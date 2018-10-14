// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <assert.h>


#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#include <avxintrin.h>
#include <immintrin.h>
#endif



namespace bb {

inline float bb_mm256_cvtss_f32(__m256 a)
{
#ifdef _MSC_VER
	return a.m256_f32[0];
#else
	return a[0];
//	return _mm256_cvtss_f32(a);
#endif
}

inline __m256 bb_mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
{
#ifdef __AVX2__
	return _mm256_fmadd_ps(a, b, c);
#else
	return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
#endif
}

inline __m256 bb_mm256_fmsub_ps(__m256 a, __m256 b, __m256 c)
{
#ifdef __AVX2__
	return _mm256_fmsub_ps(a, b, c);
#else
	return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

inline __m256 bb_mm256_fnmadd_ps(__m256 a, __m256 b, __m256 c)
{
#ifdef __AVX2__
	return _mm256_fnmadd_ps(a, b, c);
#else
	return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}


inline __m256i bb_mm256_andnot_si256(__m256i a, __m256i b)
{
#ifdef __AVX2__
	return _mm256_andnot_si256(a, b);
#else
	__m256 res = _mm256_andnot_ps(*(__m256 *)&a, *(__m256 *)&b);
	return *(__m256i *)&res;
#endif
}

inline __m256i bb_mm256_and_si256(__m256i a, __m256i b)
{
#ifdef __AVX2__
	return _mm256_and_si256(a, b);
#else
	__m256 res = _mm256_and_ps(*(__m256 *)&a, *(__m256 *)&b);
	return *(__m256i *)&res;
#endif
}

inline __m256i bb_mm256_or_si256(__m256i a, __m256i b)
{
#ifdef __AVX2__
	return _mm256_or_si256(a, b);
#else
	__m256 res = _mm256_or_ps(*(__m256 *)&a, *(__m256 *)&b);
	return *(__m256i *)&res;
#endif
}

}

#ifndef __AVX2__
#define _mm256_andnot_si256		bb_mm256_andnot_si256
#define _mm256_and_si256		bb_mm256_and_si256
#define _mm256_or_si256			bb_mm256_or_si256
#define _mm256_fmadd_ps			bb_mm256_fmadd_ps
#define _mm256_fmsub_ps			bb_mm256_fmsub_ps
#define _mm256_fnmadd_ps		bb_mm256_fnmadd_ps
#endif


