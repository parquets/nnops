#pragma once
namespace nnops::backend::cpu {

void FregPackRhs(bool trans, int Mr, int Kc, float* dst, const float* src, int ld, float scale);
void TilePackRhs(bool trans, int Mr, int Kc, float* dst, const float* src, int ld, float scale);

void FregPackLhs(bool trans, int Mr, int Kc, float* dst, const float* src, int ld, float scale);
void TilePackLhs(bool trans, int Mr, int Kc, float* dst, const float* src, int ld, float scale);

void TileMmaPack(float* c, int ldc, const float* a, const float* b);
void TileMmaDirect();

}  // namespace nnops::backend::cpu
