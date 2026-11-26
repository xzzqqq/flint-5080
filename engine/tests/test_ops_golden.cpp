#include <cuda_fp16.h>
#include <linux/if_alg.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "loader/arenas.h"
#include "loader/safetensors.h"
#include "model/visit_plan.h"
#include "ops/attention_ops.h"
#include "ops/moe_ops.h"
#include "ops/norm_ops.h"
#include "ops/sconv_ops.h"
#include "ops/submodules.h"

namespace {
using namespace flint;
void Require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}
void Check(int status) {
  Require(status == 0, cudaGetErrorString(static_cast<cudaError_t>(status)));
}
template <class T>
class Buffer {
 public:
  explicit Buffer(size_t count) : n(count) {
    void* p;
    Check(cudaMalloc(&p, n * sizeof(T)));
    ptr.reset(static_cast<T*>(p));
  }
  explicit Buffer(const std::vector<T>& src) : Buffer(src.size()) {
    Check(cudaMemcpy(get(), src.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  T* get() const { return ptr.get(); }
  std::vector<T> read() const {
    std::vector<T> dst(n);
    Check(cudaMemcpy(dst.data(), get(), n * sizeof(T), cudaMemcpyDeviceToHost));
    return dst;
  }
  void zero() { Check(cudaMemset(get(), 0, n * sizeof(T))); }

 private:
  size_t n;
  std::unique_ptr<T, detail::DeviceFree> ptr;
};
float BFloat(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
uint16_t B(float x) {
  unsigned u = std::bit_cast<unsigned>(x);
  return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
struct Fixture {
  SafetensorsFile file;
  std::map<std::string, Buffer<std::byte>> device;
  const TensorEntry& entry(const std::string& name) const {
    auto e = file.Find(name);
    Require(e != nullptr, "missing " + name);
    return *e;
  }
  template <class T = void>
  const T* data(const std::string& name) {
    auto bytes = file.Bytes(entry(name));
    auto [it, inserted] = device.try_emplace(name, bytes.size());
    if (inserted)
      Check(cudaMemcpy(it->second.get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    return reinterpret_cast<const T*>(it->second.get());
  }
  template <class T>
  std::vector<T> host(const std::string& name) const {
    auto bytes = file.Bytes(entry(name));
    Require(bytes.size() % sizeof(T) == 0, "invalid size " + name);
    std::vector<T> result(bytes.size() / sizeof(T));
    memcpy(result.data(), bytes.data(), bytes.size());
    return result;
  }
};
bool alignment_ok = true;
void Compare(const std::string& name, const std::vector<float>& actual,
             const std::vector<float>& ref, double max_limit, double rms_limit) {
  Require(!ref.empty() && ref.size() == actual.size(), "shape " + name);
  double mx = 0, ma = 0, ss = 0, xx = 0, yy = 0, xy = 0;
  size_t exact = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double x = actual[i], y = ref[i], d = std::abs(x - y);
    Require(std::isfinite(d), "nonfinite " + name);
    mx = std::max(mx, d);
    ma += d;
    ss += d * d;
    xx += x * x;
    yy += y * y;
    xy += x * y;
    exact += std::bit_cast<uint32_t>(actual[i]) == std::bit_cast<uint32_t>(ref[i]);
  }
  double rmse = sqrt(ss / ref.size()), cos = xx && yy ? xy / sqrt(xx * yy) : double(xx == yy);
  std::cout << name << " " << std::setprecision(9) << mx << " " << ma / ref.size() << " " << rmse
            << " " << cos << " " << exact << "/" << ref.size() << std::endl;
  if (max_limit == 0) Require(exact == ref.size(), "bitwise mismatch " + name);
  if (!(mx <= max_limit && rmse <= rms_limit && cos >= 0.99999)) {
    std::cerr << "tolerance " << name << std::endl;
    alignment_ok = false;
  }
}
void CompareBF(Fixture& f, const std::string& name, const Buffer<uint16_t>& buf, double mx,
               double rms) {
  auto x = buf.read(), y = f.host<uint16_t>(name);
  std::vector<float> a(x.size()), b(y.size());
  std::transform(x.begin(), x.end(), a.begin(), BFloat);
  std::transform(y.begin(), y.end(), b.begin(), BFloat);
  Compare(name, a, b, mx, rms);
}
std::string Hex(const uint8_t* p, size_t n) {
  std::ostringstream out;
  for (size_t i = 0; i < n; ++i) out << std::hex << std::setfill('0') << std::setw(2) << int(p[i]);
  return out.str();
}
std::string Digest(const std::vector<uint16_t>& data) {
  struct Socket {
    int fd;
    ~Socket() {
      if (fd >= 0) close(fd);
    }
  };
  Socket hash{socket(AF_ALG, SOCK_SEQPACKET, 0)};
  sockaddr_alg address{};
  address.salg_family = AF_ALG;
  strcpy(reinterpret_cast<char*>(address.salg_type), "hash");
  strcpy(reinterpret_cast<char*>(address.salg_name), "sha256");
  Require(bind(hash.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
          "bind sha256");
  Socket op{accept(hash.fd, nullptr, nullptr)};
  const auto* bytes = reinterpret_cast<const char*>(data.data());
  size_t size = data.size() * 2;
  for (size_t i = 0; i < size;) {
    size_t chunk = std::min<size_t>(65536, size - i);
    auto sent = send(op.fd, bytes + i, chunk, MSG_MORE);
    Require(sent > 0, "sha256 update");
    i += sent;
  }
  Require(send(op.fd, nullptr, 0, 0) == 0, "sha256 finish");
  uint8_t digest[32];
  Require(read(op.fd, digest, 32) == 32, "sha256 digest");
  return Hex(digest, 32);
}
void Exl(Fixture& f, const Gemm& g, ScratchArenaView ws) {
  for (auto group : {"exl3.3_0", "exl3.3_5", "exl3.6_0"}) {
    std::string p = std::string(group) + ".";
    int k = f.entry(p + "suh").shape.at(0), n = f.entry(p + "svh").shape.at(0),
        k2 = f.entry(p + "trellis").shape.at(2) / 8;
    Require(f.host<unsigned>(p + "mul1").at(0) == FLINT_EXL3_MUL1_MAGIC, "mul1");
    Exl3MatrixView mat{f.data<int16_t>(p + "trellis"),
                       f.data<uint16_t>(p + "suh"),
                       f.data<uint16_t>(p + "svh"),
                       (unsigned)k,
                       (unsigned)n,
                       (uint8_t)k2};
    Buffer<uint16_t> inner(k * n);
    Check(flint_exl3_decode_inner_f16({mat.trellis, inner.get(), k, n, k2}, nullptr));
    auto decoded = inner.read();
    auto digest = Digest(decoded);
    auto expected = f.host<uint8_t>(p + "inner_sha256");
    Require(digest == Hex(expected.data(), expected.size()), p + "SHA mismatch " + digest);
    std::cout << p << "inner_sha256 " << digest << " exact\n";
    auto slice = f.host<uint16_t>(p + "inner_f16_slice");
    for (int i = 0; i < 32; ++i)
      Require(
          std::equal(slice.begin() + i * 32, slice.begin() + (i + 1) * 32, decoded.begin() + i * n),
          p + "slice");
    std::cout << p << "inner_f16_slice 0 0 0 1 1024/1024\n";
    int t = f.entry(p + "x").shape.at(0);
    Buffer<float> out(t * n);
    RunExl3Linear(g, ws, mat, f.data(p + "x"), out.get(), t, FLINT_F32, FLINT_F32);
    Compare(p + "y", out.read(), f.host<float>(p + "y"), 4e-6, 8e-7);
  }
}
void Golden(const char* dir, const char* file, const Gemm& g) {
  Fixture f{SafetensorsFile::Open(file), {}};
  auto model = WeightsArena::Load(dir);
  const auto& m = model.view();
  auto kv = KvStateArena::Allocate(m.config, 16);
  kv.Reset();
  auto ws = ScratchArena::Allocate();
  auto plan = BuildVisitPlan(m.config);
  // The first historical block is the reference's post-embedding RMS output.
  auto prompt = f.host<int>("input_ids");
  Buffer<uint16_t> embed(prompt.size() * 2304);
  Check(flint_embed_rmsnorm(
      {m.embed_weight, m.embed_norm_weight, embed.get(), f.data<int>("input_ids"),
       static_cast<int>(prompt.size()), 2304, FLINT_BF16, FLINT_BF16},
      g.stream()));
  CompareBF(f, "block_residual.layer_1.block", embed, 0, 0);
  Exl(f, g, ws.view());
  for (int layer : {0, 5}) {
    const auto& l = m.layers[layer];
    bool full = layer == 5;
    Buffer<float> state(3 * 1024);
    state.zero();
    std::vector<float> history(3 * 1024);
    int pos = 0;
    for (std::string phase : {"prefill", "decode_0", "decode_1"}) {
      std::string p = (full ? "dynamic_sconv." : "static_sconv.") + phase + ".";
      int t = f.entry(p + "x").shape.at(0);
      auto ids = phase == "prefill" ? prompt : f.host<int>(phase + ".input_ids");
      Require(ids.size() == size_t(t), "conv phase length");
      for (int id : ids)
        Require(id >= 0 && unsigned(id) < m.config.unpadded_vocab_size, "invalid trace token");
      Buffer<uint16_t> out(t * 1024), delta(t * 32);
      ConvParams conv{f.data(p + "x"),
                      full ? l.k_static_w : l.k_sconv,
                      delta.get(),
                      state.get(),
                      out.get(),
                      t,
                      1024,
                      FLINT_BF16,
                      FLINT_BF16};
      if (full) {
        RunRowMatmul(g, ws.view(), f.data(p + "condition"), l.k_weight_proj, delta.get(), t, 32,
                     2304, FLINT_BF16, FLINT_BF16, true);
        Check(flint_dynamic_sconv(conv, nullptr));
      } else
        Check(flint_static_sconv(conv, nullptr));
      CompareBF(f, p + "y", out, 0, 0);
      auto x = f.host<uint16_t>(p + "x");
      for (int row = 0; row < t; ++row)
        for (int c = 0; c < 1024; ++c) {
          auto h = history.data() + 3 * c;
          h[0] = h[1];
          h[1] = h[2];
          h[2] = BFloat(x[row * 1024 + c]);
        }
      Require(state.read() == history, p + "state update");

      if (phase != "decode_1") {
        p = (full ? "full_attention." : "sliding_attention.") + phase + ".";
        Buffer<uint16_t> y(t * 2304);
        AttentionBatch batch{f.data(p + "q"), f.data(p + "k"), f.data(p + "v"), y.get(), t, pos,
                             plan[layer]};
        (full ? RunFullAttention : RunSlidingAttention)(g, ws.view(), l, kv.view(), batch);
        CompareBF(f, p + "y", y, 0, 0);
      }
      pos += t;
    }
  }
  int t = f.entry("moe.x").shape.at(0);
  Buffer<uint16_t> y(t * 2304);
  Buffer<float> logits(t * 128), weights(t * 8);
  Buffer<int> ids(t * 8);
  RunSparseMoeBlock(g, ws.view(), m.layers[0], f.data("moe.x"), y.get(), t, logits.get(), ids.get(),
                    weights.get());
  Compare("moe.router_logits", logits.read(), f.host<float>("moe.router_logits"), 0, 0);
  auto ref_ids = f.host<int>("moe.top8_ids"), actual_ids = ids.read();
  Require(actual_ids == ref_ids, "moe.top8_ids slot mismatch");
  std::cout << "moe.top8_ids exact " << ref_ids.size() << "\n";
  Compare("moe.weights", weights.read(), f.host<float>("moe.weights"), 0, 0);
  CompareBF(f, "moe.y", y, 0, 0);
  for (int layer : {1, 5, 6}) {
    std::string p = layer == 6 ? "message_residual.layer_6_pass_1."
                               : "block_residual.layer_" + std::to_string(layer) + ".";
    int queries = layer == 1 ? 1 : layer == 5 ? 3 : 4;
    Buffer<float> out(t * queries * 2304), merged(4 * 2304);
    const float* q = m.layers[layer].attn_query;
    if (layer == 6) {
      Check(flint_merge_cross_pass_queries(
          {m.cross_pass_queries, m.cross_pass_kv_queries, merged.get(), 0, 2304}, nullptr));
      q = merged.get();
    }
    std::vector<float> qhost(queries * 2304);
    Check(cudaMemcpy(qhost.data(), q, qhost.size() * 4, cudaMemcpyDeviceToHost));
    Require(qhost == f.host<float>(p + "query"), p + "query order");
    RunBlockResidual(g, ws.view(),
                     {f.data(p + "block"), f.data(p + "partial"), q, out.get(), 1, t, 2304, queries,
                      FLINT_BF16, FLINT_BF16});
    Compare(p + "y", out.read(), f.host<float>(p + "y"), 1e-5, 1.2e-7);
  }
  int j = 0;
  for (std::string name : {"attn", "k", "v", "mlp"}) {
    std::string p = "message_gate." + name + ".";
    Buffer<float> out(t * 2304);
    RunMessageGate(g, ws.view(), m.layers[6].controllers[j++], f.data<float>(p + "x"),
                   f.data<float>(p + "message"), out.get(), t);
    Compare(p + "y", out.read(), f.host<float>(p + "y"), 0, 0);
  }
}

size_t FreeBytes() {
  Check(cudaDeviceSynchronize());
  size_t free, total;
  Check(cudaMemGetInfo(&free, &total));
  return free;
}

void SyntheticExl() {
  for (int k2 : {6, 7, 12}) {
    std::vector<uint32_t> words(4 * k2);
    uint32_t seed = 5080;
    for (auto& w : words) {
      seed = 1664525u * seed + 1013904223u;
      w = seed;
    }
    Buffer<uint32_t> packed(words);
    Buffer<uint16_t> inner(256);
    Check(flint_exl3_decode_inner_f16(
        {reinterpret_cast<const short*>(packed.get()), inner.get(), 16, 16, k2}, nullptr));
    auto values = inner.read();
    for (int p = 0; p < 256; ++p) {
      uint32_t state = 0;
      for (int bit = (p + 1) * k2 / 2 - 16; bit < (p + 1) * k2 / 2; ++bit) {
        int index = (bit + 128 * k2) % (128 * k2);
        state = (state << 1) | ((words[index / 32] >> (31 - index % 32)) & 1);
      }
      uint32_t product = state * 0x83dcd12du;
      int sum = 1024;
      for (int j = 0; j < 4; ++j) sum += (product >> (8 * j)) & 255;
      float value =
          sum * __half2float(__ushort_as_half(0x1eee)) + __half2float(__ushort_as_half(0xc931));
      int lane = p / 8, j = p % 8, r = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1),
          c = lane / 4 + 8 * (j / 4);
      Require(values[r * 16 + c] == __half_as_ushort(__float2half(value)), "circular decoder");
    }
  }
  std::vector<float> x(256);
  x[17] = 2;
  x[128 + 65] = -3;
  std::vector<uint16_t> scales(256);
  for (int i = 0; i < 256; ++i) scales[i] = __half_as_ushort(__float2half(float(1 + i % 3)));
  Buffer<float> input(x), rot(256), result(256);
  Buffer<uint16_t> scale(scales);
  Check(flint_wht128_scale_in({input.get(), scale.get(), rot.get(), 1, 256, FLINT_F32, FLINT_F32},
                              nullptr));
  auto y = rot.read();
  for (int i = 0; i < 256; ++i) {
    int j = i < 128 ? 17 : 65;
    float v = i < 128 ? 6 : -6;
    float ref = v * (std::popcount(unsigned(i & j)) % 2 ? -1 : 1) / sqrtf(128);
    Require(fabsf(y[i] - ref) < 1e-6, "WHT signs and scale");
  }
  Check(flint_wht128_scale_out({rot.get(), scale.get(), result.get(), 1, 256, FLINT_F32, FLINT_F32},
                               nullptr));
  y = result.read();
  for (int i = 0; i < 256; ++i)
    Require(fabsf(y[i] - x[i] * (1 + i % 3) * (1 + i % 3)) < 5e-5, "WHT output scale");
}
void SyntheticNorm() {
  std::vector<uint16_t> host{B(3), B(4), B(5), B(-12)}, scale{B(2), B(0.5)};
  Buffer<uint16_t> x(host), w(scale), out(4);
  Buffer<int> ids(std::vector<int>{1, 0});
  Check(flint_embed_rmsnorm({x.get(), w.get(), out.get(), ids.get(), 2, 2, FLINT_BF16, FLINT_BF16},
                            nullptr));
  auto y = out.read();
  for (int i = 0; i < 4; ++i) {
    int j = (i + 2) % 4;
    float v = BFloat(host[j]);
    float ss = i < 2 ? 84.5f : 12.5f;
    Require(y[i] == B(v / sqrtf(ss + 1e-6f) * BFloat(scale[i % 2])), "embedding RMS");
  }
  Buffer<float> f(std::vector<float>{3, 4}), z(2);
  Check(flint_rmsnorm({f.get(), w.get(), z.get(), nullptr, 1, 2, FLINT_F32, FLINT_F32}, nullptr));
  auto v = z.read();
  Require(fabsf(v[0] - 6 / sqrtf(12.5f + 1e-6f)) < 1e-6 &&
              fabsf(v[1] - 2 / sqrtf(12.5f + 1e-6f)) < 1e-6,
          "weighted RMS");
  Check(flint_unweighted_rmsnorm({f.get(), nullptr, z.get(), nullptr, 1, 2, FLINT_F32, FLINT_F32},
                                 nullptr));
  Require(fabsf(z.read()[1] - 4 / sqrtf(12.5f + 1e-6f)) < 1e-6, "unweighted RMS");
}
void SyntheticConv() {
  for (bool dynamic : {false, true}) {
    int c = dynamic ? 1024 : 2304;
    std::vector<float> x(6 * c);
    for (int i = 0; i < 6 * c; ++i) x[i] = float(i / c + 1) * (1 + i % 7);
    std::vector<uint16_t> weights(c * 4), delta(6 * 32);
    for (int i = 0; i < c; ++i)
      for (int lag = 0; lag < 4; ++lag)
        weights[dynamic ? lag * c + i : 4 * i + 3 - lag] = B((lag - 1) * (1 + i % 3));
    for (int i = 0; i < 6 * 32; ++i) delta[i] = B(float(i % 13 - 6) * 0.25f);
    Buffer<float> input(x), out(6 * c), state(c * 3);
    state.zero();
    Buffer<uint16_t> w(weights), d(delta);
    int pos = 0;
    for (int t : {4, 1, 1}) {
      ConvParams p{input.get() + pos * c,
                   w.get(),
                   d.get() + pos * 32,
                   state.get(),
                   out.get() + pos * c,
                   t,
                   c,
                   FLINT_F32,
                   FLINT_F32};
      Check((dynamic ? flint_dynamic_sconv : flint_static_sconv)(p, nullptr));
      pos += t;
      auto history = state.read();
      for (int i = 0; i < c; ++i)
        for (int j = 0; j < 3; ++j)
          Require(history[3 * i + j] == x[(pos - 3 + j) * c + i], "conv history");
    }
    auto y = out.read();
    for (int t = 0; t < 6; ++t)
      for (int i = 0; i < c; ++i) {
        float sum = x[t * c + i];
        for (int lag = 0; lag <= std::min(t, 3); ++lag) {
          float coeff = (lag - 1) * (1 + i % 3);
          if (dynamic) coeff += BFloat(delta[(t * 8 + i / 128) * 4 + lag]);
          sum += coeff * x[(t - lag) * c + i];
        }
        Require(fabsf(y[t * c + i] - sum / (dynamic ? 3.f : sqrtf(5.f))) < 2e-5, "conv output");
      }
  }
}

void SyntheticAttention() {
  for (int dim : {16, 128}) {
    int n = dim * 24;
    Buffer<uint16_t> input(std::vector<uint16_t>(n, B(5))), scale(std::vector<uint16_t>(n, B(2))),
        out(n);
    for (int full : {0, 1}) {
      Check((dim == 16 ? flint_rel_norm_clip : flint_q_norm_clip)(
          {input.get(), scale.get(), out.get(), 1, 16383, full}, nullptr));
      float clipped =
          BFloat(B(10.f * (dim == 16 ? 8.f : 22.627417f) / (10.f * sqrtf(float(dim)) + 1e-6f)));
      auto got = out.read();
      for (auto y : got)
        Require(y == B(clipped * (full ? 1.f + 0.1f * logf(2.f) : 1.f)), "clip and logscale");
    }
    if (dim == 128) {
      Check(flint_k_l2_clip({input.get(), nullptr, out.get(), 1, 0, 0}, nullptr));
      Require(out.read()[0] == B(2.f), "key clip");
    }
  }
  // The oldest Full key is outside the relative-bias extent. SWA overwrites
  // its slot at 1024. A large distinctive value makes either mask error visible.
  for (int sliding : {0, 1}) {
    int cap = sliding ? 1024 : 1025;
    std::vector<uint16_t> values(1025 * 1024), keys(values.size(), 0);
    for (int t = 0; t < 1025; ++t)
      for (int h = 0; h < 8; ++h) {
        keys[t * 1024 + h * 128] = B(t == 0 ? 8.f : 0.f);
        for (int c = 0; c < 128; ++c)
          values[t * 1024 + h * 128 + c] = B(t == 0 ? 4096.f : t >= 1023 ? 64.f : float(h));
      }
    Buffer<uint16_t> k(keys), v(values), ck(cap * 1024), cv(cap * 1024);
    std::vector<uint16_t> q(3072, 0), rel(24 * 1024, 0);
    for (int h = 0; h < 24; ++h) {
      q[h * 128] = B(2);
      rel[h * 1024] = B(2);
      rel[h * 1024 + 1023] = B(-1);
    }
    Buffer<uint16_t> query(q), bias(rel), sink(std::vector<uint16_t>(24, B(6))), out(3072);
    Buffer<float> scores(cap * 24);
    Check(flint_kv_cache_write({k.get(), v.get(), ck.get(), cv.get(), 1023, 0, cap, sliding},
                               nullptr));
    // Batched prefill must ignore already-written future rows.
    Buffer<uint16_t> two_q(2 * 3072), two_rel(2 * 24 * 1024), two_sink(2 * 24), two_out(2 * 3072);
    two_q.zero();
    two_rel.zero();
    two_sink.zero();
    Buffer<float> two_scores(2 * 24 * cap);
    Check(flint_attention_forward({two_q.get(), ck.get(), cv.get(), two_rel.get(), two_sink.get(),
                                   two_out.get(), two_scores.get(), 2, 0, cap, sliding},
                                  nullptr));
    auto prefill = two_out.read();
    for (int i = 0; i < 3072; ++i) {
      int kv_head = i / (3 * 128);
      Require(prefill[i] == B(2048.f) &&
                  prefill[3072 + i] == B(BFloat(B(1.f / 3)) * (4096.f + kv_head)),
              "batched causal attention");
    }
    for (int pos : {1023, 1024}) {
      Check(flint_kv_cache_write(
          {k.get() + pos * 1024, v.get() + pos * 1024, ck.get(), cv.get(), 1, pos, cap, sliding},
          nullptr));
      Check(flint_attention_forward({query.get(), ck.get(), cv.get(), bias.get(), sink.get(),
                                     out.get(), scores.get(), 1, pos, cap, sliding},
                                    nullptr));
      auto actual = out.read();
      int begin = sliding ? std::max(0, pos - 1023) : 0;
      for (int h = 0; h < 24; ++h) {
        std::vector<float> prob;
        float den = expf(6.f);
        for (int j = begin; j <= pos; ++j) {
          int d = pos - j;
          float dot = j == 0 ? BFloat(B(16.f / sqrtf(128.f))) : 0.f;
          float z = BFloat(B(dot + (d == 0 ? 2.f : d == 1023 ? -1.f : 0.f)));
          prob.push_back(expf(z));
          den += prob.back();
        }
        float ref = 0;
        for (int j = begin; j <= pos; ++j)
          ref += BFloat(B(prob[j - begin] / den)) * BFloat(values[j * 1024 + (h / 3) * 128]);
        for (int c = 0; c < 128; ++c)
          Require(actual[h * 128 + c] == B(ref), "cache wrap, GQA, distance or sink");
      }
    }
  }
  Buffer<uint16_t> in(std::vector<uint16_t>{B(2), B(3)}), gate(std::vector<uint16_t>{B(-1), B(2)}),
      out(2);
  Check(flint_gated_mul_sigmoid({in.get(), gate.get(), out.get(), 2}, nullptr));
  Require(out.read() == std::vector<uint16_t>{B(2 / (1 + expf(1))), B(3 / (1 + expf(-2)))},
          "attention gate");
}
void SyntheticGrouping() {
  constexpr int tokens = 35, rows = tokens * 8;
  int tile_rows = flint_exl3_grouped_gemm_tile_rows();
  std::vector<int> ids(rows), counts(128);
  for (int s = 0; s < rows; ++s) {
    ids[s] = (17 * (s % 8) + (s / 8 < tile_rows ? 0 : 1)) % 128;
    ++counts[ids[s]];
  }
  int max_tiles = std::min(rows, rows / tile_rows + 128);
  Buffer<int> input(ids), experts(rows), token(rows), inverse(rows), tile(max_tiles),
      begin(max_tiles), end(max_tiles), num(1);
  Check(flint_moe_group_tokens({input.get(), experts.get(), token.get(), inverse.get(), tile.get(),
                                begin.get(), end.get(), num.get(), tokens, tile_rows},
                               nullptr));
  auto group = experts.read(), source = token.read(), map = inverse.read();
  std::vector<bool> seen(rows);
  for (int s = 0; s < rows; ++s) {
    int r = map[s];
    Require(r >= 0 && r < rows && !seen[r], "group permutation");
    seen[r] = true;
    Require(group[r] == ids[s] && source[r] == s / 8, "group slot mapping");
  }
  auto e = tile.read(), b = begin.read(), z = end.read();
  int row = 0, i = 0, active = num.read()[0];
  Require(active >= 0 && active <= max_tiles, "active tile count");
  for (int id = 0; id < 128; ++id) {
    int limit = row + counts[id];
    while (row < limit) {
      int next = std::min(row + tile_rows, limit);
      Require(i < active && e[i] == id && b[i] == row && z[i] == next, "expert tile range");
      Require(std::all_of(group.begin() + row, group.begin() + next,
                          [id](int expert) { return expert == id; }),
              "expert tile contents");
      row = next;
      ++i;
    }
  }
  Require(row == rows && i == active, "tile coverage");
}
void SyntheticMoe() {
  std::vector<float> logits(128, -1), bias(128, 0);
  bias[120] = 1;
  logits[2] = 1;
  Buffer<float> l(logits), b(bias), w(8);
  Buffer<int> ids(8);
  Check(flint_moe_router_top8({l.get(), b.get(), ids.get(), w.get(), 1}, nullptr));
  auto order = ids.read();
  Require(order == std::vector<int>{2, 120, 0, 1, 3, 4, 5, 6}, "router tie break");
  float total = 0;
  for (auto id : order) total += 1 / (1 + expf(-logits[id]));
  auto weights = w.read();
  for (int i = 0; i < 8; ++i)
    Require(fabsf(weights[i] - 8 / (1 + expf(-logits[order[i]])) / total) < 5e-7,
            "unbiased router weight");
  std::vector<uint16_t> gu{B(-7), B(8), B(40), B(-30)};
  Buffer<uint16_t> input(gu), act(2);
  Check(flint_softcapped_swiglu({input.get(), act.get(), 1, 2}, nullptr));
  auto result = act.read();
  for (int i = 0; i < 2; ++i) {
    float g = BFloat(gu[i]), u = BFloat(gu[i + 2]);
    Require(result[i] == B(4 * tanhf(g / 4) / (1 + expf(-g)) * 25 * tanhf(u / 25)),
            "raw-gate sigmoid");
  }
  std::vector<uint16_t> ys(16);
  for (int i = 0; i < 16; ++i) ys[i] = B(float(i - 7));
  Buffer<uint16_t> experts(ys);
  Check(flint_moe_weighted_accum({experts.get(), w.get(), act.get(), 1, 2, ids.get(), nullptr},
                                 nullptr));
  result = act.read();
  for (int c = 0; c < 2; ++c) {
    float sum = 0;
    for (int i = 0; i < 8; ++i) sum += weights[i] * BFloat(ys[2 * i + c]);
    Require(result[c] == B(sum), "MoE accumulation");
  }
}
void SyntheticLoop(const Gemm& g) {
  auto ws = ScratchArena::Allocate(4096);
  const std::vector<float> block{1, 2, 5, -3}, partial{3, 4};
  Buffer<float> blocks(block), last(partial);
  for (int qs : {1, 3, 4}) {
    std::vector<float> query(2 * qs);
    std::iota(query.begin(), query.end(), -2.f);
    Buffer<float> q(query), out(qs * 2);
    RunBlockResidual(
        g, ws.view(),
        {blocks.get(), last.get(), q.get(), out.get(), 2, 1, 2, qs, FLINT_F32, FLINT_F32});
    auto got = out.read();
    for (int j = 0; j < qs; ++j) {
      float den = 0, sum[2]{};
      for (int i = 0; i < 3; ++i) {
        auto x = i == 2 ? partial.data() : block.data() + 2 * i;
        float inv = 1 / sqrtf((x[0] * x[0] + x[1] * x[1]) / 2 + 1e-6f);
        float weight = expf((query[2 * j] * x[0] + query[2 * j + 1] * x[1]) * inv / sqrtf(2));
        den += weight;
        for (int c = 0; c < 2; ++c) sum[c] += weight * x[c] * inv;
      }
      for (int c = 0; c < 2; ++c)
        Require(fabsf(sum[c] / den - got[2 * j + c]) < 1e-6, "normalized residual states");
    }
  }
  std::vector<float> qm(24 * 2), kv(24 * 2);
  std::iota(qm.begin(), qm.end(), 0.f);
  std::iota(kv.begin(), kv.end(), 100.f);
  Buffer<float> q(qm), k(kv), merged(8);
  Check(flint_merge_cross_pass_queries({q.get(), k.get(), merged.get(), 11, 2}, nullptr));
  Require(merged.read() == std::vector<float>{44, 45, 144, 145, 146, 147, 46, 47},
          "merge query order");
  Buffer<uint16_t> prep(4);
  Check(flint_message_gate_prep({blocks.get(), last.get(), prep.get(), 1, 2}, nullptr));
  Require(prep.read() ==
              std::vector<uint16_t>{B(1 / sqrtf(2.5f + 1e-6f)), B(2 / sqrtf(2.5f + 1e-6f)),
                                    B(3 / sqrtf(12.5f + 1e-6f)), B(4 / sqrtf(12.5f + 1e-6f))},
          "message gate normalization");
  Buffer<float> out(2);
  Check(flint_message_gate_mix(
      {blocks.get(), last.get(), blocks.get() + 2, last.get(), out.get(), 2}, nullptr));
  auto got = out.read();
  for (int i = 0; i < 2; ++i)
    Require(fabsf(got[i] -
                  ((1 - tanhf(block[2 + i])) * block[i] + tanhf(partial[i]) * partial[i])) < 1e-6,
            "message mix");
}
void SyntheticComposition(const Gemm& g) {
  auto scratch = ScratchArena::Allocate(48u << 20);
  // A zero trellis decodes to a constant inner matrix. The two Hadamard
  // transforms leave only column zero of each 128-wide output block.
  Buffer<int16_t> trellis(144 * 144 * 48);
  trellis.zero();
  const uint16_t one_sixteenth = __half_as_ushort(__float2half(0.0625f));
  Buffer<uint16_t> scale(std::vector<uint16_t>(2304, one_sixteenth));
  Exl3MatrixView gate_up{trellis.get(), scale.get(), scale.get(), 2304, 2304, 6};
  Exl3MatrixView down{trellis.get(), scale.get(), scale.get(), 1152, 2304, 6};
  LayerWeightsView layer{};
  for (int e = 0; e < 128; ++e) {
    layer.gate_up_proj[e] = gate_up;
    layer.down_proj[e] = down;
  }
  Buffer<uint16_t> router(128 * 2304);
  router.zero();
  Buffer<float> bias(128);
  bias.zero();
  layer.router_weight = router.get();
  layer.router_bias = bias.get();
  std::vector<uint16_t> x(2304);
  x[0] = B(1);
  Buffer<uint16_t> input(x), output(2304);
  RunSparseMoeBlock(g, scratch.view(), layer, input.get(), output.get(), 1);
  float code = __half2float(__float2half(1024.f * __half2float(__ushort_as_half(0x1eee)) +
                                         __half2float(__ushort_as_half(0xc931))));
  float factor = code * 128 / 256;
  float gate = BFloat(B(factor));
  float act = BFloat(B(4 * tanhf(gate / 4) / (1 + expf(-gate)) * 25 * tanhf(gate / 25)));
  float expected = 8 * BFloat(B(9 * act * factor));
  auto result = output.read();
  for (int i = 0; i < 2304; ++i)
    Require(fabsf(BFloat(result[i]) - (i % 128 == 0 ? expected : 0.f)) < 1e-5f,
            "synthetic MoE projection");

  // Compare the grouped path with the original one-row cuBLAS projections.
  // Every selected expert sees every token, with distinct weights and both bitrates.
  std::vector<int16_t> codes(144 * 144 * 56 + 128 * 2);
  unsigned state = 5080;
  for (auto& word : codes) {
    state = state * 1664525u + 1013904223u;
    word = static_cast<int16_t>(state >> 16);
  }
  Buffer<int16_t> packed(codes);
  for (int e = 0; e < 128; ++e) {
    for (auto* projection : {layer.gate_up_proj, layer.down_proj}) {
      projection[e].trellis = packed.get() + 2 * e;
      projection[e].k2 = 6 + e % 2;
    }
  }
  for (int t : {5, 35}) {
    int rows = t * 8;
    std::vector<uint16_t> values(t * 2304);
    for (auto& v : values) {
      state = state * 1664525u + 1013904223u;
      v = B(float(int(state >> 24) - 128) / 128);
    }
    Buffer<uint16_t> src(values), grouped(t * 2304), ref(t * 2304), outputs(rows * 2304), gu(2304),
        activated(1152);
    Buffer<int> ids(rows);
    Buffer<float> weights(rows);
    RunSparseMoeBlock(g, scratch.view(), layer, src.get(), grouped.get(), t, nullptr, ids.get(),
                      weights.get());
    auto selected = ids.read();
    for (int s = 0; s < rows; ++s) {
      RunExl3Linear(g, scratch.view(), layer.gate_up_proj[selected[s]], src.get() + (s / 8) * 2304,
                    gu.get(), 1);
      Check(flint_softcapped_swiglu({gu.get(), activated.get(), 1, 1152}, g.stream()));
      RunExl3Linear(g, scratch.view(), layer.down_proj[selected[s]], activated.get(),
                    outputs.get() + s * 2304, 1);
    }
    Check(flint_moe_weighted_accum(
        {outputs.get(), weights.get(), ref.get(), t, 2304, ids.get(), nullptr}, g.stream()));
    Require(grouped.read() == ref.read(), "grouped MoE vs scalar t=" + std::to_string(t));
  }

  // The sole active rank reads the first feature of each normalized input.
  std::vector<uint16_t> d(64 * 4608), f(2304 * 64), a(2304 * 64);
  d[0] = B(2);
  d[2304] = B(0.5f);
  for (int i = 0; i < 2304; ++i) {
    f[i * 64] = B(i % 2 ? 0.25f : -0.5f);
    a[i * 64] = B(0.5f);
  }
  Buffer<uint16_t> dw(d), fw(f), aw(a);
  Buffer<float> hidden(std::vector<float>(2304, 2)), message(std::vector<float>(2304, -1)),
      mixed(2304);
  RunMessageGate(g, scratch.view(), {dw.get(), fw.get(), aw.get()}, hidden.get(), message.get(),
                 mixed.get(), 1);
  auto y = mixed.read();
  for (int i = 0; i < 2304; ++i)
    Require(fabsf(y[i] - (2 * (1 - tanhf(1.5f * BFloat(f[i * 64]))) - tanhf(0.75f))) < 2e-6,
            "synthetic controller");
}

void Synthetic(const Gemm& g) {
  SyntheticComposition(g);
  SyntheticExl();
  SyntheticNorm();
  SyntheticConv();
  SyntheticAttention();
  SyntheticGrouping();
  SyntheticMoe();
  SyntheticLoop(g);
  Buffer<float> x(std::vector<float>{1, 2, 3, 3, 100}), logits(4);
  Buffer<int> id(1);
  Check(flint_logit_softcap_argmax({x.get(), logits.get(), id.get(), 1, 4, 5, FLINT_F32}, nullptr));
  Require(id.read()[0] == 2, "softcap argmax tie");
  auto capped = logits.read();
  for (int j = 0; j < 4; ++j)
    Require(fabsf(capped[j] - 30 * tanhf(std::min(j + 1, 3) / 30.f)) < 1e-6f, "logit softcap");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 1 || (argc == 5 && std::string(argv[1]) == "--model-dir" &&
                          std::string(argv[3]) == "--golden-file"),
            "usage: ops_tests [--model-dir DIR --golden-file FILE]");
    // Initialize the driver and cuBLAS modules before measuring owned allocations.
    {
      Buffer<std::byte> workspace(4 << 20);
      flint::Gemm warm(nullptr, workspace.get(), 4 << 20);
      Buffer<float> tile(32 * 32), out(32 * 32);
      tile.zero();
      warm.Multiply(tile.get(), tile.get(), out.get(), 32, 32, 32, FLINT_F32, FLINT_F32, false);
      Check(cudaDeviceSynchronize());
    }
    size_t before = FreeBytes();
    {
      Buffer<std::byte> workspace(4 << 20);
      flint::Gemm gemm(nullptr, workspace.get(), 4 << 20);
      std::cout << "operator max_abs_err mean_abs_err rmse cos_sim exact\n";
      if (argc == 5)
        Golden(argv[2], argv[4], gemm);
      else
        Synthetic(gemm);
    }
    auto after = FreeBytes();
    std::cout << "VRAM leak bytes " << (int64_t(before) - int64_t(after)) << std::endl;
    Require(before == after, "VRAM teardown delta");
    Require(alignment_ok, "golden alignment failed");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ops_tests: " << e.what() << std::endl;
    return 1;
  }
}
