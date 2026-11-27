#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int flint_swa_kv_backup(void* ring, void* backup, int pos, int count, void* stream);
int flint_swa_kv_backup_device(void* ring, void* backup, const int32_t* d_pos, int count,
                               void* stream);
int flint_swa_kv_restore(void* ring, void* backup, int pos, int count, int accepted, void* stream);
int flint_sconv_commit_trace(const float* trace, float* state, int channels, int count,
                             int accepted, void* stream);
#ifdef __cplusplus
}
#endif
