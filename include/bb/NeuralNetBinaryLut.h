// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------



#pragma once

#include <cmath>
#include <array>
#include <vector>
#include "bb/NeuralNetSparseLayer.h"

namespace bb {


// LUT型の基本クラス
// 力技での学習を実装
// フラットな結線であれば重複が無いので、LUT単位で統計を取りながら演算が可能
// 畳み込み時はbit毎に結果に相互影響するのでbit単位でやるしか無さそう


// LUT方式基底クラス
template <bool feedback_bitwise = false, typename T = float, typename INDEX = size_t>
class NeuralNetBinaryLut : public NeuralNetSparseLayer<T, INDEX>
{
	typedef NeuralNetSparseLayer<T, INDEX> super;

protected:
	INDEX					m_frame_size = 1;

public:
	// LUT操作の定義
	virtual int   GetLutInputSize(void) const = 0;
	virtual int   GetLutTableSize(void) const = 0;
	virtual void  SetLutInput(INDEX node, int bitpos, INDEX input_node) = 0;
	virtual INDEX GetLutInput(INDEX node, int bitpos) const = 0;
	virtual void  SetLutTable(INDEX node, int bitpos, bool value) = 0;
	virtual bool  GetLutTable(INDEX node, int bitpos) const = 0;

	int   GetNodeInputSize(INDEX node) const { return GetLutInputSize(); }
	void  SetNodeInput(INDEX node, int input_index, INDEX input_node) { SetLutInput(node, input_index, input_node);  }
	INDEX GetNodeInput(INDEX node, int input_index) const { return GetLutInput(node, input_index); }


	void InitializeCoeff(std::uint64_t seed)
	{
		std::mt19937_64                     mt(seed);
		std::uniform_int_distribution<int>	rand(0, 1);
		
		INDEX node_size = this->GetOutputNodeSize();
		int   lut_input_size = GetLutInputSize();
		int   lut_table_size = GetLutTableSize();
		
		ShuffleSet	ss(this->GetInputNodeSize(), mt());
		for (INDEX node = 0; node < node_size; ++node) {
			// 入力をランダム接続
			auto random_set = ss.GetRandomSet(this->GetLutInputSize());
			for (int i = 0; i < lut_input_size; ++i) {
				this->SetLutInput(node, i, random_set[i]);
			}
			
			// LUTテーブルをランダムに初期化
			for (int i = 0; i < lut_table_size; i++) {
				this->SetLutTable(node, i, rand(mt) != 0);
			}
		}
	}
	
	template <typename RT, typename RI>
	void ImportLayer(const NeuralNetSparseLayer<RT, RI>& src)
	{
		auto node_size = this->GetOutputNodeSize();
		auto input_size = this->GetLutInputSize();
		auto table_size = this->GetLutTableSize();

		BB_ASSERT(src.GetOutputNodeSize() == node_size);
		for (INDEX node = 0; node < node_size; ++node) {
			BB_ASSERT(src.GetNodeInputSize(node) == input_size);

			// 入力をコピー
			for (int input_index = 0; input_index < input_size; ++input_index) {
				SetLutInput(node, input_index, src.GetNodeInput(node, input_index));
			}

			// 係数をバイナリ化
			std::vector<T> vec(input_size);
			for (int index = 0; index < table_size; ++index) {
				for (int bit = 0; bit < input_size; ++bit) {
					vec[bit] = (index & (1 << bit)) ? (RT)1.0 : (RT)0.0;
				}
				auto v = src.CalcNode(node, vec);
				SetLutTable(node, index, (v[0] > 0));
			}
		}
	}


public:
	bool GetLutInputSignal(INDEX frame, INDEX node, int bitpos) const
	{
		INDEX input_node = GetLutInput(node, bitpos);
		return this->GetInputSignalBuffer().template Get<bool>(frame, input_node);
	}

	virtual int GetLutInputIndex(INDEX frame, INDEX node) const
	{
		const auto& buf = this->GetInputSignalBuffer();
		int lut_input_size = GetLutInputSize();
		int index = 0;
		int mask = 1;
		for (int bitpos = 0; bitpos < lut_input_size; ++bitpos) {
			INDEX input_node = GetLutInput(node, bitpos);
			if (this->GetInputSignalBuffer().template Get<bool>(frame, input_node)) {
				index |= mask;
			}
			mask <<= 1;
		}
		return index;
	}


	void  SetBatchSize(INDEX batch_size) { m_frame_size = batch_size; }

	INDEX GetInputFrameSize(void) const { return m_frame_size; }
	INDEX GetOutputFrameSize(void) const { return m_frame_size; }

	int   GetInputSignalDataType(void) const { return BB_TYPE_BINARY; }
	int   GetInputErrorDataType(void) const { return NeuralNetType<T>::type; }
	int   GetOutputSignalDataType(void) const { return BB_TYPE_BINARY; }
	int   GetOutputErrorDataType(void) const { return NeuralNetType<T>::type; }

protected:
	virtual void ForwardNode(INDEX node)
	{
		auto in_sig_buf = this->GetInputSignalBuffer();
		auto out_sig_buf = this->GetOutputSignalBuffer();
		int   lut_input_size = GetLutInputSize();

		for (INDEX frame = 0; frame < m_frame_size; ++frame) {
			int index = 0;
			int mask = 1;
			for (int i = 0; i < lut_input_size; i++) {
				INDEX input_node = GetLutInput(node, i);
				bool input_signal = in_sig_buf.template Get<bool>(frame, input_node);
				index |= input_signal ? mask : 0;
				mask <<= 1;
			}
			bool output_signal = GetLutTable(node, index);
			out_sig_buf.template Set<bool>(frame, node, output_signal);
		}
	}

public:
	virtual void Forward(bool train = true)
	{
		INDEX node_size = this->GetOutputNodeSize();
		int   lut_input_size = this->GetLutInputSize();

		#pragma omp parallel for
		for ( int node = 0; node < (int)node_size; ++node) {
			ForwardNode(node);
		}
	}
		
	void Backward(void)
	{
		auto& out_err = this->GetOutputErrorBuffer();
		auto& in_err = this->GetInputErrorBuffer();

		INDEX frame_size = this->GetOutputFrameSize();
		INDEX node_size = this->GetOutputNodeSize();
		int lut_input_size = this->GetLutInputSize();
		int lut_table_size = this->GetLutTableSize();

		// ゼロ初期化
		INDEX input_node_size = this->GetInputNodeSize();
		for (INDEX node = 0; node < input_node_size; ++node) {
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				in_err.template Set<T>(frame, node, 0);
			}
		}

		std::mt19937_64 mt(1);

		// 計算
		std::vector<T> table_err(lut_table_size);
		for (INDEX node = 0; node < node_size; ++node) {
			std::fill(table_err.begin(), table_err.end(), (T)0);
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				// 入力値取得
				int input_index = this->GetLutInputIndex(frame, node);
				T err = out_err.template Get<T>(frame, node);

				// テーブルに対する誤差計算
				table_err[input_index] += err;	// 積算していく
			}

			for (int bitpos = 0; bitpos < lut_input_size; ++bitpos) {
				if ( std::abs(table_err[bitpos]) > (mt() % 16)+5 ) {
					this->SetLutTable(node, bitpos, table_err[bitpos] > 0);
				}
			}
			
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				int input_index = this->GetLutInputIndex(frame, node);
				T err = out_err.template Get<T>(frame, node);

				bool val = GetLutTable(node, input_index);
				if ((val && err < 0) || (val && err > 0)) {

					// 入力に対する伝播誤差計算
					int mask = 1;
			//		for (int bitpos = 0; bitpos < lut_input_size; ++bitpos) {
					{
						int bitpos = (int)(mt() % lut_input_size);

						INDEX input_node = GetLutInput(node, bitpos);
						// 各入力項に対するテーブルの偏微分を計算
						int index0 = (input_index & ~mask);
						int index1 = (input_index | mask);
						bool val0 = this->GetLutTable(node, index0);
						bool val1 = this->GetLutTable(node, index1);

						if (!val0 && val1) {
							in_err.template Set<T>(frame, input_node, in_err.template Get<T>(frame, input_node) + err);
						}
						else if (val0 && !val1) {
							in_err.template Set<T>(frame, input_node, in_err.template Get<T>(frame, input_node) - err);
						}
						mask <<= 1;
					}
				}
			}

		}
	}
	
	
	void Update(double learning_rate)
	{
	}


protected:
	inline int GetLutInputIndex(NeuralNetBuffer<T, INDEX>& buf, int lut_input_size, INDEX frame, INDEX node)
	{
		// 入力値作成
		int index = 0;
		int mask = 1;
		for (int i = 0; i < lut_input_size; ++i) {
			INDEX input_node = GetLutInput(node, i);
			index |= (buf.template Get<bool>(frame, input_node) ? mask : 0);
			mask <<= 1;
		}
		return index;
	}


	// feedback
	std::mt19937_64						m_feedback_mt;
	bool								m_feedback_busy = false;
	INDEX								m_feedback_node;
	int									m_feedback_bit;
	int									m_feedback_phase;
	std::vector< std::vector<int> >		m_feedback_input;
	std::vector<double>					m_feedback_loss;
	
	// 入力を集計してLUT単位で学習
	inline bool FeedbackLutwise(const std::vector<double>& loss)
	{
		auto in_buf = this->GetInputSignalBuffer();
		auto out_buf = this->GetOutputSignalBuffer();

		INDEX node_size = this->GetOutputNodeSize();
		INDEX frame_size = this->GetOutputFrameSize();
		int lut_input_size = this->GetLutInputSize();
		int	lut_table_size = this->GetLutTableSize();

		// 初回設定
		if (!m_feedback_busy) {
			m_feedback_busy = true;
			m_feedback_node = 0;
			m_feedback_phase = 0;
			m_feedback_loss.resize(lut_table_size);

			m_feedback_input.resize(node_size);
			for (INDEX node = 0; node < node_size; ++node) {
				m_feedback_input[node].resize(frame_size);
				for (INDEX frame = 0; frame < frame_size; ++frame) {
					// 入力値作成
#if 1
					int value = 0;
					int mask = 1;
					for (int i = 0; i < lut_input_size; ++i) {
						INDEX input_node = GetLutInput(node, i);
						value |= (in_buf.template Get<bool>(frame, input_node) ? mask : 0);
						mask <<= 1;
					}
					m_feedback_input[node][frame] = value;
#else
					m_feedback_input[node][frame] = GetLutInputIndex(frame, node);
#endif
				}
			}
		}

		// 完了
		if (m_feedback_node >= node_size) {
			m_feedback_busy = false;
			return false;
		}

		if ( m_feedback_phase == 0 ) {
			// 結果を集計
			std::fill(m_feedback_loss.begin(), m_feedback_loss.end(), (T)0.0);
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				int lut_input = m_feedback_input[m_feedback_node][frame];
				m_feedback_loss[lut_input] += loss[frame];
			}

			// 出力を反転
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				out_buf.template Set<bool>(frame, m_feedback_node, !out_buf.template Get<bool>(frame, m_feedback_node));
			}

			m_feedback_phase++;
		}
		else {
			// 反転させた結果を集計
			for (INDEX frame = 0; frame < frame_size; ++frame) {
				int lut_input = m_feedback_input[m_feedback_node][frame];
				m_feedback_loss[lut_input] -= loss[frame];
			}

			// 集計結果に基づいてLUTを学習
			int	lut_table_size = GetLutTableSize();
			for (int bit = 0; bit < lut_table_size; ++bit) {
				std::normal_distribution<double> rand_distribution(0, 0.001);

	//			if ( abs(m_feedback_loss[bit]) >= abs(rand_distribution(m_feedback_mt)) ) {
					if (m_feedback_loss[bit] > 0) {
						SetLutTable(m_feedback_node, bit, !GetLutTable(m_feedback_node, bit));
					}
	//			}
			}

			// 学習したLUTで出力を再計算
			ForwardNode(m_feedback_node);

			// 次のLUTに進む
			m_feedback_phase = 0;
			++m_feedback_node;
		}

		return true;	// 以降を再計算して継続
	}

	// [アイデアだけメモ]
	// 畳み込みでビット単位はあまりうまくいっていない
	// ハミング距離1づつしか移動できないので局所解に落ちやすい模様
	// 例えば4bitぐらい束にして16回まわして一番良いものを取るとかが
	// 必要な気がする。選ぶ4bitは異なるLUTを跨ぐのもありだと思う

	// ビット単位で学習
	inline bool FeedbackBitwise(const std::vector<double>& loss)
	{
		auto in_buf = this->GetInputSignalBuffer();
		auto out_buf = this->GetOutputSignalBuffer();

		INDEX node_size = this->GetOutputNodeSize();
		INDEX frame_size = this->GetOutputFrameSize();
		int lut_input_size = this->GetLutInputSize();
		int	lut_table_size = this->GetLutTableSize();

		// 初回設定
		if (!m_feedback_busy) {
			m_feedback_busy = true;
			m_feedback_node = 0;
			m_feedback_bit  = 0;
			m_feedback_phase = 0;
			m_feedback_loss.resize(1);
		}

		// 完了
		if (m_feedback_node >= node_size) {
			m_feedback_busy = false;
			return false;
		}

		// 損失集計
		double loss_sum = (T)0;
		for (auto v : loss) {
			loss_sum += v;
		}

		if (m_feedback_phase == 0) {
			// 損失を保存
			m_feedback_loss[0] = loss_sum;

			// 該当LUTを反転
			SetLutTable(m_feedback_node, m_feedback_bit, !GetLutTable(m_feedback_node, m_feedback_bit));

			// 変更したLUTで再計算
			ForwardNode(m_feedback_node);

			++m_feedback_phase;
		}
		else {
			// 損失を比較
			m_feedback_loss[0] -= loss_sum;

			std::normal_distribution<double> dist(0.0, 0.1);

			if (m_feedback_loss[0] < 0) {
				// 反転させない方が結果がよければ元に戻す
				SetLutTable(m_feedback_node, m_feedback_bit, !GetLutTable(m_feedback_node, m_feedback_bit));

				// 変更したLUTで再計算
				ForwardNode(m_feedback_node);
			}

			// 次のbitに進む
			m_feedback_phase = 0;
			++m_feedback_bit;

			if (m_feedback_bit >= lut_table_size) {
				// 次のbitLUTに進む
				m_feedback_bit = 0;
				++m_feedback_node;
			}
		}

		return true;	// 以降を再計算して継続
	}


public:
	bool Feedback(const std::vector<double>& loss)
	{
		if (feedback_bitwise) {
			return FeedbackBitwise(loss);
		}
		else {
			return FeedbackLutwise(loss);
		}
	}

	// Serialize
protected:
	struct LutData {
		std::vector<INDEX>	lut_input;
		std::vector<bool>	lut_table;

		template <class Archive>
		void serialize(Archive &archive, std::uint32_t const version)
		{
			archive(cereal::make_nvp("input", lut_input));
			archive(cereal::make_nvp("table", lut_table));
		}
	};

public:
	template <class Archive>
	void save(Archive &archive, std::uint32_t const version) const
	{
		archive(cereal::make_nvp("NeuralNetLayer", *(super *)this));

		archive(cereal::make_nvp("input_node_size", this->m_input_node_size));
		archive(cereal::make_nvp("m_output_node_size", this->m_output_node_size));

		INDEX node_size = this->GetOutputNodeSize();
		int lut_input_size = this->GetLutInputSize();
		int	lut_table_size = this->GetLutTableSize();

		std::vector<LutData> vec_lut;
		for (INDEX node = 0; node < node_size; ++node) {
			LutData ld;
			ld.lut_input.resize(lut_input_size);
			for (int i = 0; i < lut_input_size; ++i) {
				ld.lut_input[i] = this->GetLutInput(node, i);
			}

			ld.lut_table.resize(lut_table_size);
			for (int i = 0; i < lut_table_size; ++i) {
				ld.lut_table[i] = this->GetLutTable(node, i);
			}

			vec_lut.push_back(ld);
		}

		archive(cereal::make_nvp("lut", vec_lut));
	}
	

	template <class Archive>
	void load(Archive &archive, std::uint32_t const version)
	{
		archive(cereal::make_nvp("NeuralNetLayer", *(super *)this));
		
		INDEX input_node_size;
		INDEX output_node_size;
		std::vector<LutData> vec_lut;
		archive(cereal::make_nvp("input_node_size", input_node_size));
		archive(cereal::make_nvp("m_output_node_size", output_node_size));
		archive(cereal::make_nvp("lut", vec_lut));

//		if (vec_lut.empty()) { return; }

		this->Resize(input_node_size, output_node_size);

		for (INDEX node = 0; node < (INDEX)vec_lut.size(); ++node) {
			for (int i = 0; i < (int)vec_lut[node].lut_input.size(); ++i) {
				SetLutInput(node, i, vec_lut[node].lut_input[i]);
			}

			for (int i = 0; i < (int)vec_lut[node].lut_table.size(); ++i) {
				SetLutTable(node, i, vec_lut[node].lut_table[i]);
			}
		}
	}

	virtual void Save(cereal::JSONOutputArchive& archive) const
	{
		archive(cereal::make_nvp("NeuralNetBinaryLut", *this));
	}

	virtual void Load(cereal::JSONInputArchive& archive)
	{
		archive(cereal::make_nvp("NeuralNetBinaryLut", *this));
	}
};

}

