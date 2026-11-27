#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int flint_swa_kv_backup(void* ring, void* backup, int pos, int count, void* stream);
int flint_swa_kv_restore(void* ring, void* backup, int pos, int count, int accepted, void* stream);
int flint_sconv_commit_trace(const float* trace, float* state, int channels, int count,
                             int accepted, void* stream);
#ifdef __cplusplus
}
#endif
