// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <string.h>
#include <memory>
#include <atomic>
#include <type_traits>

#ifdef BB_WITH_CUDA
#include "cuda_runtime.h"
#include "bbcu/bbcu.h"
#endif

#include "bb/DataType.h"
#include "bb/Utility.h"
#include "bb/CudaUtility.h"


namespace bb {


//[Memory クラス]
//  ・GPU/CPUどちらのメモリも管理
//  ・コピータイミングを明示的に管理したいのと、 Maxwell以前のGPUも使いたいので Unified Memory は一旦保留
//  ・Lcok/Unlockで管理
//  ・Lock した段階で、自分のデバイスに最新データが無ければcudaMemcopy
//
//
// メモリの実体を管理する Memory クラスと
// Memory のロックを管理する Ptr_ / ConstPtr_ クラス とからなる
// 参照が存在している間、実体やロックが維持される
// Memory クラスから Ptr を取得することで、実メモリがアクセス可能な状態でロックされ
// Ptrの生存期間が過ぎるとロック解除される
// ロックしなおした際にアドレスが変わらない保証は行わない

class Memory
{
public:
    // メモリポインタ(const)
    template <void (*lock)(Memory*), void (*unlock)(Memory*)>
    class ConstPtr_
    {
        friend Memory;

    protected:
        void const *m_addr = nullptr;
        Memory     *m_mem = nullptr;

        inline void Lock()    const { lock(m_mem); }
        inline void Unlock()  const { if (m_mem != nullptr) { unlock(m_mem); } }

    protected:
       // friend の Memoryクラスのみ初期値を与えられる
        ConstPtr_(void const *addr, Memory *mem) noexcept
        {
            m_addr = addr;
            m_mem = mem;
            Lock();
        }

    public:
        ConstPtr_() {}

        ConstPtr_(ConstPtr_ const &obj)
        {
            Unlock();
            m_addr = obj.m_addr;
            m_mem = obj.m_mem;
            Lock();
        }

        ~ConstPtr_()
        {
            Unlock();
        }

        ConstPtr_& operator=(ConstPtr_ const &obj)
        {
            Unlock();
            m_addr = obj.m_addr;
            m_mem = obj.m_mem;
            Lock();
            return *this;
        }

        bool IsEmpty(void) const
        {
            return (m_mem == nullptr);
        }

        void Clear(void)
        {
            Unlock();
            m_addr = nullptr;
            m_mem = nullptr;
        }
        
        void const* GetAddr(void) const
        {
            return m_addr;
        }

        template<typename Tp>
        Tp const& At(index_t index) const {
//          BB_DEBUG_ASSERT(m_ptr != nullptr);
            return ((Tp const*)m_addr)[index];
        }
    };

    
    // メモリポインタ
    template <typename ConstTp, void (*lock)(Memory*), void (*unlock)(Memory*)>
    class Ptr_
    {
        friend Memory;

    protected:
        void*   m_addr = nullptr;
        Memory* m_mem  = nullptr;

        inline void Lock()    const { lock(m_mem); }
        inline void Unlock()  const { if (m_mem != nullptr) { unlock(m_mem); } }

    protected:
        // friend の Memoryクラスのみ初期値を与えられる
        Ptr_(void* addr, Memory* mem)
        {
            m_addr = addr;
            m_mem  = mem;
            Lock();
        }

    public:
        Ptr_() {}

        Ptr_(Ptr_ const &obj)
        {
            Unlock();
            m_addr = obj.m_addr;
            m_mem = obj.m_mem;
            Lock();
        }

        ~Ptr_()
        {
            Unlock();
        }

        Ptr_& operator=(Ptr_ const &obj)
        {
            Unlock();
            m_addr = obj.m_addr;
            m_mem = obj.m_mem;
            Lock();
            return *this;
        }

        bool IsEmpty(void) const
        {
            return (m_mem == nullptr);
        }

        void Clear(void)
        {
            Unlock();
            m_addr = nullptr;
            m_mem = nullptr;
        }
        
        void* GetAddr(void) const
        {
            return m_addr;
        }

        operator ConstTp() const
        {
           return ConstTp(m_addr, m_mem);
        }

        // constアクセス
        template<typename Tp>
        Tp const & At(index_t index) const {
//          BB_DEBUG_ASSERT(m_ptr != nullptr);
            return ((Tp const *)m_addr)[index];
        }

        // 非constアクセス
        template<typename Tp>
        Tp& At(index_t index) {
//          BB_DEBUG_ASSERT(m_ptr != nullptr);
            return ((Tp *)m_addr)[index];
        }
    };

protected:
    void*               m_addr = nullptr;
    size_t              m_size = 0;
    std::atomic<int>    m_hostRefCnt;
    bool                m_hostOnly = true;
    bool                m_hostModified = false;

#ifdef BB_WITH_CUDA
    size_t              m_mem_size = 0;
    bool                m_devAvailable = false;

    // 将来下記を多重化して複数GPU対応もケアできるようにするかも
    int                 m_device = 0;
    void*               m_devAddr = nullptr;
    bool                m_devModified = false;
    std::atomic<int>    m_devRefCnt;
#endif

#ifdef BB_WITH_CUDA
    static void GetRef(Memory *self)       { BB_ASSERT(self->m_devRefCnt == 0);  self->m_hostRefCnt++; }
    static void RelRef(Memory *self)       { BB_ASSERT(self->m_devRefCnt == 0);  self->m_hostRefCnt--; }
    static void GetRefDevice(Memory *self) { BB_ASSERT(self->m_hostRefCnt == 0); self->m_devRefCnt++; }
    static void RelRefDevice(Memory *self) { BB_ASSERT(self->m_hostRefCnt == 0); self->m_devRefCnt--; }
#else
    static void GetRef(Memory *self)       { self->m_hostRefCnt++; }
    static void RelRef(Memory *self)       { self->m_hostRefCnt--; }
    static void GetRefDevice(Memory *self) {}
    static void RelRefDevice(Memory *self) {}
#endif

public:
    using ConstPtr    = ConstPtr_<GetRef, RelRef>;                        //< 読み書き可能なHOSTメモリのポインタオブジェクト
    using Ptr         = Ptr_<ConstPtr, GetRef, RelRef>;                   //< リードオンリーなHOSTメモリのポインタオブジェクト
    using DevConstPtr = ConstPtr_<&GetRefDevice, &RelRefDevice>;          //< 読み書き可能なDeviceメモリのポインタオブジェクト
    using DevPtr      = Ptr_<DevConstPtr, &GetRefDevice, &RelRefDevice>;  //< リードオンリーなDeviceTメモリのポインタオブジェクト

    friend Ptr;
    friend ConstPtr;
    friend DevPtr;
    friend DevConstPtr;


public:
    /**
     * @brief  メモリオブジェクトの生成
     * @detail メモリオブジェクトの生成
     * @param size 確保するメモリサイズ(バイト単位)
     * @param device 利用するGPUデバイス
     *           0以上  現在の選択中のGPU
     *           -1     現在の選択中のGPU
     *           -2     GPUは利用しない
     * @return メモリオブジェクトへのshared_ptr
     */
    static std::shared_ptr<Memory> Create(size_t size, bool hostOnly=false)
    {
        return std::shared_ptr<Memory>(new Memory(size, hostOnly));
    }

protected:
    /**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     * @param size 確保するメモリサイズ(バイト単位)
     * @param device 利用するGPUデバイス
     *           0以上  現在の選択中のGPU
     *           -1     現在の選択中のGPU
     *           -2     GPUは利用しない
     * @return なし
     */
    Memory(size_t size, bool hostOnly=false) : m_hostRefCnt(0)
    {
        // 初期化
        m_size       = size;
        m_hostRefCnt = 0;
        m_hostOnly   = hostOnly;

#ifdef BB_WITH_CUDA
        m_mem_size     = m_size;
        m_devRefCnt    = 0;
        m_devAvailable = false;

        // デバイス設定
        int dev_count = bbcu_GetDeviceCount();

        // デバイスがあれば有効化
        if ( dev_count > 0 && !m_hostOnly ) {
            m_devAvailable = true;
            m_device = bbcu_GetDevice();
        }

        // デバイスが使えなければここでホストメモリ確保
        if ( !m_devAvailable ) {
            m_addr = aligned_memory_alloc(m_size, 32);
        }
#else
        // メモリ確保
        m_addr = aligned_memory_alloc(m_size, 32);
#endif
    }

public:
    /**
     * @brief  デストラクタ
     * @detail デストラクタ
     */
    ~Memory()
    {
        BB_DEBUG_ASSERT(m_hostRefCnt == 0);

#ifdef BB_WITH_CUDA
        BB_DEBUG_ASSERT(m_devRefCnt == 0);

        if ( m_devAvailable ) {
            CudaDevicePush dev_push(m_device);

            // Hostメモリ開放
            if (m_addr != nullptr) {
                bbcu::FreeHost(m_addr);
            }

            // Deviceメモリ開放
            if (m_devAddr != nullptr) {
                bbcu::Free(m_devAddr);
            }
        }
        else {
            // メモリ開放
            if (m_addr != nullptr) {
                aligned_memory_free(m_addr);
            }
        }

#else
        // メモリ開放
        if (m_addr != nullptr) {
            aligned_memory_free(m_addr);
        }
#endif
    }

   /**
     * @brief  メモリオブジェクトの生成
     * @detail メモリオブジェクトの生成
     * @param size 確保するメモリサイズ(バイト単位)
     * @param device 利用するGPUデバイス
     *           0以上  現在の選択中のGPU
     *           -1     現在の選択中のGPU
     *           -2     GPUは利用しない
     * @return メモリオブジェクトへのshared_ptr
     */
    std::shared_ptr<Memory> Clone(void) const
    {
#ifdef BB_WITH_CUDA
        auto clone = std::shared_ptr<Memory>(new Memory(m_size, m_hostOnly));

        if (m_addr == nullptr && m_devAddr == nullptr) {
            return clone;
        }

        if (m_hostModified || !IsDeviceAvailable() || clone->IsDeviceAvailable() ) {
            auto ptr_src = LockConst();
            auto ptr_dst = clone->Lock(true);
            memcpy(ptr_dst.GetAddr(), ptr_src.GetAddr(), m_size);
        }
        else {
            // ひとまず複数GPUは未サポート
            int device;
            BB_CUDA_SAFE_CALL(cudaGetDevice(&device));
            BB_ASSERT(device == m_device);

            auto ptr_src = LockDeviceConst();
            auto ptr_dst = clone->LockDevice(true);
            bbcu::Memcpy(ptr_dst.GetAddr(), ptr_src.GetAddr(), m_size, cudaMemcpyDeviceToDevice);
        }
        return clone;
#else
        auto clone = std::shared_ptr<Memory>(new Memory(m_size));
        memcpy(clone->m_addr, m_addr, m_size);
        return clone;
#endif        
    }
    

    /**
     * @brief  メモリのサイズを変更する
     * @detail メモリのサイズを変更する
     *         古い中身はサイズに関わらず破棄する
     */
    void Resize(size_t size)
    {
        BB_ASSERT(m_hostRefCnt == 0);

#ifdef BB_WITH_CUDA
        BB_ASSERT(m_devRefCnt == 0);
        m_size = size;
        if ( m_devAvailable ) {
            // デバイスメモリ再確保
            if (m_size <= m_mem_size) {
                return;
            }
            if (m_addr != nullptr) {
                bbcu::FreeHost(m_addr);  // Hostメモリ開放
                m_addr= nullptr;
            }
            if (m_devAddr != nullptr) {
                bbcu::Free(m_devAddr);   // Deviceメモリ開放
                m_devAddr= nullptr;
            }
            m_mem_size = m_size;
            m_hostModified = false;
            m_devModified = false;
        }
        else {
            // ホストメモリ再確保
            aligned_memory_free(m_addr);
            m_addr = aligned_memory_alloc(size, 32);
            m_hostModified = false;
        }
#else
        aligned_memory_free(m_addr);
        m_addr = aligned_memory_alloc(size, 32);
        m_size = size;
        m_hostModified = false;
#endif
    }


    /**
     * @brief  メモリサイズの取得
     * @detail メモリサイズの取得
     * @return メモリサイズ(バイト単位)
     */
    index_t GetSize(void) const
    {
        return m_size;
    }

    /**
     * @brief  ホスト専用かどうか問い合わせる
     * @detail ホスト専用かどうか問い合わせる
     * @return ホスト専用ならtrue
     */
    bool IsHostOnly(void) const
    {
#ifdef BB_WITH_CUDA
        return m_hostOnly;
#else
        return true;
#endif
    }

    /**
     * @brief  デバイスが利用可能か問い合わせる
     * @detail デバイスが利用可能か問い合わせる
     * @return デバイスが利用可能ならtrue
     */
    bool IsDeviceAvailable(void) const
    {
#ifdef BB_WITH_CUDA
        return m_devAvailable;
#else
        return false;
#endif
    }
    
    /**
     * @brief  ゼロ初期化する
     * @detail ゼロ初期化する
     */
    void FillZero(void)
    {
        if ( m_size == 0 ) { return; }

#ifdef BB_WITH_CUDA
        // メモリ未確保なら確保
        if (m_addr == nullptr && m_devAddr == nullptr) {
            Lock(true);
        }
        
        // クリア
        if (m_addr != nullptr) {
            memset(m_addr, 0, m_size);
        }
        m_hostModified = false;

        if (m_devAddr != nullptr) {
            BB_CUDA_SAFE_CALL(cudaMemset(m_devAddr, 0, m_size));
        }
        m_devModified  = false;
#else
        // メモリ未確保なら確保
        if (m_addr == nullptr ) {
            Lock(true);
        }

        // クリア
        memset(m_addr, 0, m_size);
        m_hostModified = false;
#endif
    }

    /**
     * @brief  メモリ内容の破棄
     * @detail メモリ内容を破棄する
     */ void Dispose(void)
    {
        // 更新の破棄
        m_hostModified = false;

#ifdef BB_WITH_CUDA
        m_devModified = false;
#endif
    }

    /**
     * @brief  ポインタの取得
     * @detail アクセス用に確保したホスト側のメモリポインタの取得
     * @param  new_buffer true なら古い内容を破棄する
     * @return ホスト側のメモリポインタ
     */
    Ptr Lock(bool new_buffer=false)
    {
#ifdef BB_WITH_CUDA
        if ( m_devAvailable ) {
            // 新規であれば過去の更新情報は破棄
            if ( new_buffer ) {
                m_hostModified = false;
                m_devModified = false;
            }

            if (m_addr == nullptr) {
                // ホスト側メモリ未確保ならここで確保
                CudaDevicePush dev_push(m_device);
                bbcu::MallocHost(&m_addr, m_mem_size);
            }

            if ( m_devModified ) {
                // デバイス側メモリが最新ならコピー取得
                CudaDevicePush dev_push(m_device);
                bbcu::Memcpy(m_addr, m_devAddr, m_size, cudaMemcpyDeviceToHost);
                m_devModified =false;
            }
        }
#endif

        // 修正フラグセット
        m_hostModified = true;

        // ポインタオブジェクトを生成して返す
        return Ptr(m_addr, this);
    }

    /**
     * @brief  読み取り専用ポインタの取得
     * @detail アクセス用に確保したホスト側のメモリポインタの取得
     *         実際にはメモリのロックなどで内部状態が変わるが、
     *         メモリ内容が変わらないので便宜上 const とする
     * @return アクセス用に確保したホスト側のメモリポインタ
     */
    ConstPtr LockConst(void) const
    {
        auto self = const_cast<Memory *>(this);

#ifdef BB_WITH_CUDA
        if ( m_devAvailable ) {
            if (m_addr == nullptr) {
                // ホスト側メモリ未確保ならここで確保
                CudaDevicePush dev_push(m_device);
                bbcu::MallocHost(&self->m_addr, m_mem_size);
            }

            if ( m_devModified ) {
                // デバイス側メモリが最新ならコピー取得
                CudaDevicePush dev_push(m_device);
                bbcu::Memcpy(m_addr, m_devAddr, m_size, cudaMemcpyDeviceToHost);
                self->m_devModified = false;
            }
        }
#endif

        // ポインタを生成して返す
        return ConstPtr(m_addr, self);
    }


    /**
     * @brief  デバイス側ポインタの取得
     * @detail アクセス用に確保したデバイス側のメモリポインタの取得
     * @param  new_buffer true なら古い内容を破棄する
     * @return デバイス側のメモリポインタ
     */
    DevPtr LockDevice(bool new_buffer=false)
    {
    #ifdef BB_WITH_CUDA
        if ( m_devAvailable ) {
            // 新規であれば過去の更新情報は破棄
            if (new_buffer) {
                m_hostModified = false;
                m_devModified = false;
            }

            if (m_devAddr == nullptr) {
                // デバイス側メモリ未確保ならここで確保
                CudaDevicePush dev_push(m_device);
                bbcu::Malloc(&m_devAddr, m_size);
            }

            if (m_hostModified) {
                // ホスト側メモリが最新ならコピー取得
                CudaDevicePush dev_push(m_device);
                bbcu::Memcpy(m_devAddr, m_addr, m_size, cudaMemcpyHostToDevice);
                m_hostModified =false;
            }

            // 修正フラグセット
            m_devModified = true;

            return DevPtr(m_devAddr, this);
        }
#endif

        BB_ASSERT(0);
        return DevPtr();    // エラー
    }


    /**
     * @brief  デバイス側ポインタの取得
     * @detail アクセス用に確保したデバイス側のメモリポインタの取得
     * @param  new_buffer true なら古い内容を破棄する
     * @return デバイス側のメモリポインタ
     */
    DevConstPtr LockDeviceConst(void) const
    {
#ifdef BB_WITH_CUDA
        // 便宜上constをはずす
        auto self = const_cast<Memory *>(this);

        if ( m_devAvailable ) {
            if (m_devAddr == nullptr) {
                // デバイス側メモリ未確保ならここで確保
                CudaDevicePush dev_push(m_device);
                bbcu::Malloc(&self->m_devAddr, m_size);
            }

            if (m_hostModified) {
                // ホスト側メモリが最新ならコピー取得
                CudaDevicePush dev_push(m_device);
                bbcu::Memcpy(m_devAddr, m_addr, m_size, cudaMemcpyHostToDevice);
                self->m_hostModified =false;
            }

            return DevConstPtr(m_devAddr, self);
        }
#endif

        BB_ASSERT(0);
        return DevConstPtr();    // エラー
    }


    /**
     * @brief  hostOnlyフラグの変更
     * @detail hostOnlyフラグの変更
     * @param  新しいhostOnlyフラグの
     */
    void SetHostOnly(bool hostOnly)
    {
        BB_ASSERT(m_hostRefCnt == 0);

#ifdef BB_WITH_CUDA
        BB_ASSERT(m_devRefCnt == 0);

        // 変更が無ければ何もしない
        if ( hostOnly ==  m_hostOnly ) {
            return;
        }

        if (hostOnly) {
            // メモリ確保
            auto newAddr = aligned_memory_alloc(m_size, 32);
            BB_ASSERT(m_addr != nullptr);

            // データがあればコピー
            if ( m_hostModified ) {
                memcpy(newAddr, m_addr, m_size);
            }
            else if ( m_devModified ) {
                bbcu::Memcpy(newAddr, m_devAddr, m_size, cudaMemcpyDeviceToHost);
            }

            // デバイスメモリ開放
            if (m_addr != nullptr) {
                bbcu::FreeHost(m_addr);  // Hostメモリ開放
                m_addr= nullptr;
            }
            if (m_devAddr != nullptr) {
                bbcu::Free(m_devAddr);   // Deviceメモリ開放
                m_devAddr= nullptr;
            }
            m_mem_size = m_size;
            m_hostModified = false;
            m_devModified = false;

            m_addr = newAddr;
        }
        else {
            if ( m_hostModified ) {
                // メモリ確保
                void *newAddr;
                bbcu::MallocHost(&newAddr, m_size);
                m_mem_size = m_size;

                // コピー
                memcpy(newAddr, m_addr, m_size);

                // メモリ開放
                if ( m_addr != nullptr ) {
                    aligned_memory_free(m_addr);
                }

                m_hostModified = false;

                m_addr = newAddr;
            }
        }
#endif

        // フラグ変更
        m_hostOnly = hostOnly;
    }



    // ---------------------------------
    //  Utility
    // ---------------------------------
    
    // 3 operand
    struct Op3Ptr {
        Ptr      dst;
        ConstPtr src0;
        ConstPtr src1;
    };
    
    static Op3Ptr GetOp3Ptr(std::shared_ptr<Memory> &dst, std::shared_ptr<Memory> const &src0, std::shared_ptr<Memory> const &src1)
    {
        Op3Ptr op3;
        if ( (dst != src0) && (dst != src1) ) {
            op3.dst  = dst->Lock(true);
            op3.src0 = src0->LockConst();
            
            if ( src1 == src0 ) {
                op3.src1 = op3.src0;
            }
            else {
                op3.src1 = src1->LockConst();
            }
        }
        else {
            op3.dst = dst->Lock(false);
            
            if (src0 == dst) {
                op3.src0 = op3.dst;
            }
            else {
                op3.src0 = src0->LockConst();
            }

            if (src1 == dst) {
                op3.src1 = op3.dst;
            }
            else {
                op3.src1 = src1->LockConst();
            }
        }
        return op3;
    }
    
    // 3 operand
    struct DevOp3Ptr {
        DevPtr      dst;
        DevConstPtr src0;
        DevConstPtr src1;
    };
    
    static DevOp3Ptr GetDevOp3Ptr(std::shared_ptr<Memory> &dst, std::shared_ptr<Memory> const &src0, std::shared_ptr<Memory> const &src1)
    {
        DevOp3Ptr op3;
        if ( (dst != src0) && (dst != src1) ) {
            op3.dst  = dst->LockDevice(true);
            op3.src0 = src0->LockDeviceConst();
            
            if ( src1 == src0 ) {
                op3.src1 = op3.src0;
            }
            else {
                op3.src1 = src1->LockDeviceConst();
            }
        }
        else {
            op3.dst = dst->LockDevice(false);
            
            if (src0 == dst) {
                op3.src0 = op3.dst;
            }
            else {
                op3.src0 = src0->LockDeviceConst();
            }

            if (src1 == dst) {
                op3.src1 = op3.dst;
            }
            else {
                op3.src1 = src1->LockDeviceConst();
            }
        }
        return op3;
    }
};



}


// end of file
