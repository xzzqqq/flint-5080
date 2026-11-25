#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "loader/arenas.h"
#include "loader/safetensors.h"

namespace {
using namespace flint;
namespace fs = std::filesystem;
using namespace std::string_literals;
static_assert(std::is_trivial_v<ModelWeightsView> && std::is_standard_layout_v<ModelWeightsView>);
static_assert(std::is_move_constructible_v<WeightsArena> &&
              !std::is_copy_constructible_v<WeightsArena>);
void Require(bool ok, const std::string& reason) {
  if (!ok) throw std::runtime_error(reason);
}
void Cuda(cudaError_t err) { Require(err == cudaSuccess, cudaGetErrorString(err)); }
struct TempDir {
  fs::path dir;
  TempDir() {
    auto pattern = (fs::temp_directory_path() / "flint-loader-XXXXXX").string();
    Require(mkdtemp(pattern.data()) != nullptr, "mkdtemp failed");
    dir = pattern;
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};
template <class F>
void Reject(std::string_view diagnostic, F run) {
  try {
    run();
  } catch (const std::exception& ex) {
    Require(std::string_view(ex.what()).find(diagnostic) != std::string_view::npos,
            "unexpected rejection: "s + ex.what());
    return;
  }
  throw std::runtime_error("expected rejection: " + std::string(diagnostic));
}
void WriteFile(const fs::path& path, std::string_view header, std::string_view payload) {
  std::ofstream out(path, std::ios::binary);
  uint64_t len = header.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>(len >> (8 * i)));
  out << header << payload;
  Require(bool(out), "write fixture");
}
void TestFile(const fs::path& path) {
  const auto header =
      R"({"__metadata__":{"test":"\uD83D\uDE42"},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]},"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"BOOL","shape":[0],"data_offsets":[6,6]}})";
  WriteFile(path, header, "abcdef");
  {
    auto file = SafetensorsFile::Open(path);
    Require(file.file_size() == 8 + strlen(header) + 6 && file.payload_bytes() == 6 &&
                file.tensors().size() == 3 && file.metadata().at("test") == "🙂",
            "valid header");
    auto a = file.Find("a"), b = file.Find("b");
    Require(a && b && !file.Find("absent") && a->dtype == TensorDType::kF32 &&
                b->shape == std::vector<int64_t>{2},
            "tensor lookup");
    auto src = file.Bytes(*a);
    Require(src.size() == 4 && memcmp(src.data(), "abcd", 4) == 0 && file.Bytes(*b).size() == 2 &&
                memcmp(file.Bytes(*b).data(), "ef", 2) == 0 &&
                file.Bytes(*file.Find("empty")).empty(),
            "mapped bytes");
  }
  WriteFile(path, header, "");
  fs::resize_file(path, 15);
  Reject("header length", [&] { SafetensorsFile::Open(path); });
  auto invalid = [&](auto json, auto payload, auto error) {
    WriteFile(path, json, payload);
    Reject(error, [&] { SafetensorsFile::Open(path); });
  };
  invalid("{", "", "invalid JSON");
  invalid("{\"__metadata__\":{\"x\":\"\xc0\xaf\"}}", "", "invalid JSON");
  invalid(R"({"x":1,"x":2})", "", "invalid JSON");
  invalid(R"({"__metadata__":{"x":1}})", "", "JSON value type");
  invalid(
      R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
      "x", "overlap");
  invalid(R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})", "xy", "gap");
  invalid(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,2]}})", "xy", "shape/byte");
  invalid(R"({"a":{"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,0]}})", "",
          "overflow");
  invalid(R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}})", "", "negative");
  invalid(R"({"a":{"dtype":"F32","shape":[1.2],"data_offsets":[0,0]}})", "", "numeric");
}
constexpr char kConfig[] = R"(
{
  "eos_token_id": 200006,
  "quantization_config": {
    "quant_method": "exl3",
    "codebook": "mul1"
  },
  "text_config": {
    "model_type": "inkling2_loop_text",
    "model_max_length": 1048576,
    "torch_dtype": "bfloat16",
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "vocab_size": 201024,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "d_rel": 16,
    "rel_extent": 1024,
    "q_bias": false,
    "o_bias": false,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-06,
    "use_value_norm_no_gamma": true,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [
      0,
      1,
      2,
      3,
      4,
      6,
      7,
      8,
      9,
      10,
      12,
      13,
      14,
      15,
      16,
      18,
      19,
      20,
      21,
      22
    ],
    "sconv_kernel_size": 4,
    "unpadded_vocab_size": 200058,
    "final_logit_softcapping": 30.0,
    "swa_head_dim": 128,
    "swa_num_attention_heads": 24,
    "swa_num_key_value_heads": 8,
    "sliding_window_size": 1024,
    "n_layers_per_attn_res_block": 6,
    "attn_res_normalize_values": true,
    "n_routed_experts": 128,
    "num_experts_per_tok": 8,
    "n_shared_experts": 0,
    "intermediate_size": 1152,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": true,
    "loop_config": {
      "core_start": 6,
      "core_end": 18,
      "num_passes": 3,
      "gate_rank": 64,
      "is_varloop": false
    }
  }
}
)";
ModelConfig TestConfig(const fs::path& path) {
  auto bad = std::string(kConfig);
  auto pos = bad.find("\"is_varloop\": false");
  bad.replace(pos, strlen("\"is_varloop\": false"), "\"is_varloop\": true");
  std::ofstream(path) << bad;
  Reject("is_varloop", [&] { LoadModelConfig(path); });
  std::ofstream(path) << kConfig;
  auto config = LoadModelConfig(path);
  Require(config.unpadded_vocab_size == 200058 && config.pad_token_id == 199999 &&
              config.loop_gate_rank == 64 && config.rms_norm_eps == 1e-6f,
          "config fields");
  auto plan = BuildVisitPlan(config);
  unsigned swa = 0, full = 0;
  for (unsigned i = 0; i < plan.size(); ++i) {
    const auto& v = plan[i];
    unsigned layer = i < 18 ? i : i < 42 ? 6 + (i - 18) % 12 : i - 24;
    unsigned pass = i >= 18 && i < 42 ? 1 + (i - 18) / 12 : 0;
    bool global = layer % 6 == 5;
    Require(v.visit_idx == i && v.physical_layer == layer && v.pass_idx == pass &&
                v.has_messages == (pass != 0) && v.is_block_boundary == (layer % 6 == 0) &&
                v.is_full_attention == global && v.kv_slot_idx == (global ? full++ : swa++),
            "incorrect visit " + std::to_string(i));
  }
  Require(swa == 40 && full == 8, "visit counts");
  return config;
}
void TestBadWeights(const fs::path& dir) {
  auto path = dir / "model.safetensors";
  auto check = [&](auto header, auto payload, auto reason) {
    WriteFile(path, header, payload);
    Reject(reason, [&] { WeightsArena::Load(dir); });
  };
  check("{}", "", "missing tensor");
  check(R"({"0.self_attn.q_proj.linear.mul1":{"dtype":"I32","shape":[],"data_offsets":[0,4]}})",
        "1234", "bad mul1 magic");
  check(R"({"extra":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", "x",
        "unexpected tensor extra");
  check(R"({"0.experts.gate_up_proj.1.suh":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})",
        "xx", "wrong dtype or shape");
}
size_t FreeBytes() {
  size_t free, total;
  Cuda(cudaMemGetInfo(&free, &total));
  return free;
}
void Aligned(const void* p) {
  Require(p && reinterpret_cast<uintptr_t>(p) % 256 == 0, "unaligned arena pointer");
}
void TestArenas(const ModelConfig& config) {
  size_t before = FreeBytes();
  {
    auto kv = KvStateArena::Allocate(config, 256);
    auto scratch = ScratchArena::Allocate(16ull << 20);
    auto& v = kv.view();
    Aligned(v.arena_base);
    Aligned(v.swa_kv);
    Aligned(v.full_kv);
    Aligned(v.kv_conv_state);
    Aligned(v.sublayer_conv_state);
    Aligned(scratch.view().device_ptr);
    Require(scratch.view().bytes == 16ull << 20 && v.max_full_seq_len == 256 &&
                v.arena_base == v.swa_kv && v.num_total_visits == 48 && v.conv_state_len == 3,
            "arena geometry");
    Require(KvStateArena::MaxFullSeqLenForBudget(config, v.arena_bytes) == 256 &&
                KvStateArena::MaxFullSeqLenForBudget(config, v.arena_bytes - 1) == 255 &&
                KvStateArena::MaxFullSeqLenForBudget(config, 0) == 0,
            "inverse budget");
    Cuda(cudaMemset(v.arena_base, 0xab, v.arena_bytes));
    kv.Reset();
    std::vector<unsigned char> buffer(4ull << 20);
    for (size_t offset = 0; offset < v.arena_bytes; offset += buffer.size()) {
      size_t n = std::min(buffer.size(), v.arena_bytes - offset);
      Cuda(cudaMemcpy(buffer.data(), static_cast<char*>(v.arena_base) + offset, n,
                      cudaMemcpyDeviceToHost));
      Require(std::all_of(buffer.begin(), buffer.begin() + n, [](auto x) { return x == 0; }),
              "KV reset");
    }
  }
  Require(FreeBytes() == before, "unit arena VRAM leak");
}

void Compare(const SafetensorsFile& file, const char* name, const void* device) {
  Aligned(device);
  auto* entry = file.Find(name);
  Require(entry != nullptr, "missing readback tensor "s + name);
  auto source = file.Bytes(*entry);
  std::vector<std::byte> buf(std::min<size_t>(8ull << 20, source.size()));
  for (size_t pos = 0; pos < source.size(); pos += buf.size()) {
    auto n = std::min(buf.size(), source.size() - pos);
    Cuda(cudaMemcpy(buf.data(), static_cast<const std::byte*>(device) + pos, n,
                    cudaMemcpyDeviceToHost));
    Require(memcmp(buf.data(), source.data() + pos, n) == 0, "readback mismatch: "s + name);
  }
  std::cout << "readback " << name << " " << source.size() << " B OK\n";
}
void TestModel(const fs::path& dir) {
  const size_t before = FreeBytes();
  size_t peak_free = 0;
  {
    auto start = std::chrono::steady_clock::now();
    auto weights = WeightsArena::Load(dir);
    double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto& w = weights.view();
    Require(w.mapped_tensor_count == 25582 && w.verified_mul1_count == 6264, "real tensor counts");
    auto scratch = ScratchArena::Allocate();
    const size_t kv_budget = FreeBytes();
    auto kv = KvStateArena::Allocate(w.config, 65536);
    peak_free = FreeBytes();
    Require(before >= kv_budget && before - kv_budget >= w.arena_bytes + scratch.view().bytes &&
                kv_budget >= peak_free && kv_budget - peak_free >= kv.view().arena_bytes,
            "device memory accounting");
    auto file = SafetensorsFile::Open(dir / "model.safetensors");
    auto payload = file.payload_bytes() - sizeof(uint32_t) * w.verified_mul1_count;
    std::cout << std::fixed << std::setprecision(6) << "mapped=" << w.mapped_tensor_count
              << " markers=" << w.verified_mul1_count
              << " uploaded=" << w.mapped_tensor_count - w.verified_mul1_count
              << " file_B=" << file.file_size() << " uploaded_B=" << payload
              << " weights_B=" << w.arena_bytes << " kv_B=" << kv.view().arena_bytes
              << " scratch_B=" << scratch.view().bytes << " wall_s=" << seconds
              << " effective_GBps=" << payload / seconds / 1e9
              << " H2D_ms=" << weights.transfer_ms()
              << " H2D_GBps=" << w.arena_bytes / (weights.transfer_ms() * 1e6)
              << " max_tokens_3550_MiB="
              << KvStateArena::MaxFullSeqLenForBudget(w.config, 3550ull << 20)
              << " available_KV_budget_B=" << kv_budget << " max_tokens_available="
              << KvStateArena::MaxFullSeqLenForBudget(w.config, kv_budget) << "\n";
    Compare(file, "model.llm.embed.weight", w.embed_weight);
    Compare(file, "model.llm.unembed.weight", w.unembed_weight);
    Compare(file, "0.self_attn.q_proj.linear.trellis", w.layers[0].q_proj.trellis);
    Compare(file, "0.experts.gate_up_proj.0.trellis", w.layers[0].gate_up_proj[0].trellis);
    Compare(file, "0.experts.gate_up_proj.1.trellis", w.layers[0].gate_up_proj[1].trellis);
    Compare(file, "model.llm.layers.5.attn.k_sconv.static_w", w.layers[5].k_static_w);
    Compare(file, "model.llm.attn_res.attn_queries.4", w.layers[5].attn_query);
    Compare(file, "model.llm.layers.6.controllers.attn.down_weight",
            w.layers[6].controllers[0].down_weight);
    Compare(file, "model.llm.middle_cycle_cross_pass_queries", w.cross_pass_queries);
    Compare(file, "model.llm.layers.0.mlp.gate.weight", w.layers[0].router_weight);
    Compare(file, "model.llm.layers.0.attn_norm.weight", w.layers[0].attn_norm);
    Compare(file, "model.llm.layers.0.attn.rel_logits_proj.proj", w.layers[0].rel_logits_proj);
    Compare(file, "0.experts.down_proj.1.svh", w.layers[0].down_proj[1].svh);
    Require(!w.layers[0].attn_query && !w.layers[0].controllers[0].down_weight &&
                !w.layers[5].attn_norm && !w.layers[0].q_input_norm,
            "nonapplicable bindings");
  }
  const size_t after = FreeBytes();
  std::cout << "free_before_B=" << before << " free_at_peak_B=" << peak_free
            << " free_after_B=" << after << "\n";
  Require(before == after, "real arena VRAM leak");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const char* dir = std::getenv("FLINT_TEST_MODEL_DIR");
    const char* golden = std::getenv("FLINT_TEST_GOLDEN_FILE");
    for (int i = 1; i < argc; ++i) {
      std::string_view opt = argv[i];
      Require(++i < argc, "missing option value");
      if (opt == "--model-dir")
        dir = argv[i];
      else if (opt == "--golden-file")
        golden = argv[i];
      else
        throw std::runtime_error("unknown option " + std::string(opt));
    }
    TempDir temp;
    TestFile(temp.dir / "test.safetensors");
    auto config = TestConfig(temp.dir / "config.json");
    TestBadWeights(temp.dir);
    Cuda(cudaFree(nullptr));
    TestArenas(config);
    if (dir) TestModel(dir);
    if (golden) {
      auto file = SafetensorsFile::Open(golden);
      Require(!file.tensors().empty(), "empty golden file");
      std::cout << "golden tensors=" << file.tensors().size() << " file_B=" << file.file_size()
                << " payload_B=" << file.payload_bytes() << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
