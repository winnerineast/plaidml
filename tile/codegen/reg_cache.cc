// Copyright 2018, Intel Corporation

#include "tile/codegen/reg_cache.h"

#include "base/util/throw.h"
#include "tile/math/bignum.h"
#include "tile/stripe/stripe.h"
#include "tile/codegen/alias.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace math;    // NOLINT
using namespace stripe;  // NOLINT

struct RegisterPassOptions {
  RefDir dir;
  Location local_loc;
  Location reg_loc;
  size_t align_size;
  size_t reg_size;
  size_t gmem_lat;
  size_t lmem_lat;
  size_t reg_lat;
  std::string comp_parent_tag;
  bool cache_index_order;
};

// Get the outer and inner (outer's first sub-block) loop count
void OuterInnerLoopCount(Block* outer, size_t* outer_loop, size_t* inner_loop) {
  *outer_loop = outer->idxs_product();
  Block* inner = outer->SubBlock(0).get();
  *inner_loop = inner->idxs_product();
}

void ClearAccesses(const Refinement& ref) {
  for (auto& access : ref.mut().access) {
    access = Affine();
  }
}

void FixRefLoc(const Refinement& src, const Refinement& dst) {
  dst.mut().location = src.location;
  dst.mut().offset = src.offset;
  for (size_t i = 0; i < dst.interior_shape.dims.size(); i++) {
    dst.mut().interior_shape.dims[i].stride = src.interior_shape.dims[i].stride;
  }
}

// Propagate the location, offset, and stride information recursively
void PropagateRefLoc(Block* block, const Refinement& outer_ref) {
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& ref : inner->refs) {
        if (ref.from == outer_ref.into()) {
          FixRefLoc(outer_ref, ref);
          PropagateRefLoc(inner.get(), ref);
        }
      }
    }
  }
}

// Copy index from src to dst except for wo
std::set<std::string> CopyIndexWithout(Block* dst, Block* src, const std::set<std::string>& wo) {
  dst->idxs.clear();
  std::set<std::string> removed;
  for (auto& idx : src->idxs) {
    if (idx.affine == Affine()) {
      dst->idxs.push_back(idx);
    }
    else {
      bool has_acc_idx = false;
      auto& acc_map = idx.affine.getMap();
      for (auto& kvp : acc_map) {
        if (kvp.first.size() > 0 && wo.find(kvp.first) != wo.end()) {
          has_acc_idx = true;
          break;
        }
      }
      if (has_acc_idx) {
        removed.insert(idx.name);
      }
      else {
        dst->idxs.push_back(idx);
      }
    }
  }
  return removed;
}

// Replace the cache block (global <-> local) with a new block (global <-> register).
// The new block's index and access reference the computing block.
void ReferenceComputingBlock(const AliasMap& parent_map,          //
                             Block* parent, Block* comp_parent,   //
                             Block* cache, Block* comp,           //
                             const std::string& ref_name,         //
                             const RegisterPassOptions& opt) {
  auto cache_inner = cache->SubBlock(0);
  auto comp_inner = comp->SubBlock(0);

  size_t load_size = comp_inner->exterior_shape(ref_name).sizes_product_bytes();
  if (load_size > opt.reg_size) {
    return;
  }

  size_t cache_oloop;
  size_t cache_iloop;
  size_t comp_oloop;
  size_t comp_iloop;
  OuterInnerLoopCount(cache, &cache_oloop, &cache_iloop);
  OuterInnerLoopCount(comp, &comp_oloop, &comp_iloop);
  if (cache_iloop * 32 <= comp_iloop) {
    return;
  }

  const auto& cache_outer_local_ref = cache->ref_by_from(ref_name);
  const auto& cache_inner_local_ref = cache_inner->ref_by_from(cache_outer_local_ref->into());
  Refinement* cache_outer_global_ref_ptr = opt.dir == RefDir::In ? cache->ref_ins()[0] : cache->ref_outs(true)[0];
  const auto& cache_outer_global_ref_orig = cache->ref_by_into(cache_outer_global_ref_ptr->into());
  const auto& cache_inner_global_ref_orig = cache_inner->ref_by_from(cache_outer_global_ref_orig->into());
  const auto& comp_outer_local_ref = comp->ref_by_into(ref_name);
  const auto& comp_inner_local_ref = comp_inner->ref_by_into(comp_outer_local_ref->into());
  std::string global_ref_name = cache_outer_global_ref_ptr->from;
  auto global_ref_shape = cache_outer_global_ref_ptr->interior_shape;

  if (parent == comp_parent) {
    cache->idxs = comp->idxs;
    cache_inner->idxs = comp_inner->idxs;
    cache_inner->constraints = comp_inner->constraints;
  }
  else {
    // Correct the affine index in cache block from computing block
    std::set<std::string> mid_idxs;
    for (auto& idx : comp_parent->idxs) {
      if (idx.affine == Affine()) {
        mid_idxs.insert(idx.name);
      }
    }
    std::set<std::string> invalid_idxs = CopyIndexWithout(cache, comp, mid_idxs);
    CopyIndexWithout(cache_inner.get(), comp_inner.get(), invalid_idxs);
    for (auto& idx : cache->idxs) {
      if (idx.affine == Affine()) {
        continue;
      }
      std::map<std::string, Affine> replacement;
      const auto& acc_map = idx.affine.getMap();
      for (const auto& kvp : acc_map) {
        if (kvp.first.size() > 0) {
          Index* comp_parent_var = comp_parent->idx_by_name(kvp.first);
          if (comp_parent_var->affine != Affine()) {
            // The index should not in middle block
            replacement.emplace(kvp.first, comp_parent_var->affine);
          }
        }
      }
      idx.affine.substitute(replacement);
    }
    // Remove the invalid constraints
    cache_inner->constraints.clear();
    for (auto& cons : comp_inner->constraints) {
      auto& acc_map = cons.getMap();
      bool valid = true;
      for (auto& kvp : acc_map) {
        if (kvp.first.size() > 0) {
          Index *idx = cache_inner->idx_by_name(kvp.first);
          if (!idx) {
            valid = false;
            break;
          }
        }
      }
      if (valid) {
        cache_inner->constraints.push_back(cons);
      }
    }
  }

  // New register refinements in cache block
  std::string rref_short_name = opt.dir == RefDir::In ? "dst" : "src";
  RefDir rref_dir = opt.dir == RefDir::In ? RefDir::Out : RefDir::In;
  Refinement cache_inner_reg_ref = comp_inner_local_ref->WithInto(rref_short_name);
  Refinement cache_outer_reg_ref = comp_outer_local_ref->WithInto(rref_short_name);
  std::vector<Extent> comp_reg_extents = comp_inner_local_ref->Extents(comp_inner->idxs);
  std::vector<size_t> comp_reg_sizes;
  for (const auto& reg_ext : comp_reg_extents) {
    comp_reg_sizes.push_back(reg_ext.max);
  }
  auto comp_reg_shape = SimpleShape(cache_inner_reg_ref.interior_shape.type,
                                    comp_reg_sizes, cache_inner_reg_ref.interior_shape.layout);
  cache_inner_reg_ref.interior_shape = comp_reg_shape;
  cache_inner_reg_ref.location.devs[0].name = "REGISTER";
  cache_inner_reg_ref.from = rref_short_name;
  cache_inner_reg_ref.dir = rref_dir;
  for (auto& dim : cache_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  cache_inner->refs.erase(cache_inner_local_ref);
  cache_inner->refs.insert(cache_inner_reg_ref);
  cache_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(cache_outer_reg_ref);
  cache_outer_reg_ref.location.devs[0].name = "REGISTER";
  cache_outer_reg_ref.dir = rref_dir;
  cache->refs.erase(cache_outer_local_ref);
  cache->refs.insert(cache_outer_reg_ref);

  // Correct global refinement access and shape
  RefDir gref_dir = opt.dir;
  Refinement cache_inner_global_ref = comp_inner_local_ref->WithInto(cache_inner_global_ref_orig->into());
  Refinement cache_outer_global_ref = comp_outer_local_ref->WithInto(cache_outer_global_ref_orig->into());
  cache_inner_global_ref.location.devs.clear();
  size_t n_dim = comp_reg_shape.dims.size();
  cache_inner_global_ref.interior_shape = global_ref_shape;
  for (size_t i = 0; i < n_dim; ++i) {
    cache_inner_global_ref.interior_shape.dims[i].size = 1;
  }
  cache_inner_global_ref.from = cache_inner_global_ref_orig->from;
  cache_inner_global_ref.dir = gref_dir;
  cache_inner_global_ref.agg_op = "";
  for (auto& dim : cache_inner_global_ref.interior_shape.dims) {
    dim.size = 1;
  }
  cache_inner->refs.erase(cache_inner_global_ref_orig);
  cache_inner->refs.insert(cache_inner_global_ref);
  cache_outer_global_ref.location.devs.clear();
  cache_outer_global_ref.interior_shape = global_ref_shape;
  for (size_t i = 0; i < n_dim; ++i) {
    cache_outer_global_ref.interior_shape.dims[i].size = comp_reg_shape.dims[i].size;
  }
  cache_outer_global_ref.from = cache_outer_global_ref_orig->from;
  cache_outer_global_ref.dir = gref_dir;
  cache_outer_global_ref.agg_op = "";
  cache->refs.erase(cache_outer_global_ref_orig);
  cache->refs.insert(cache_outer_global_ref);

  // New register refinements in computing block
  Refinement comp_inner_reg_ref = *comp_inner_local_ref;
  Refinement comp_outer_reg_ref = *comp_outer_local_ref;
  comp_inner_reg_ref.interior_shape = comp_reg_shape;
  for (auto& dim : comp_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  comp_inner_reg_ref.location.devs[0].name = "REGISTER";
  comp_inner->refs.erase(comp_inner_local_ref);
  comp_inner->refs.insert(comp_inner_reg_ref);
  comp_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(comp_outer_reg_ref);
  comp_outer_reg_ref.location.devs[0].name = "REGISTER";
  comp->refs.erase(comp_outer_local_ref);
  comp->refs.insert(comp_outer_reg_ref);

  // Add the register refinement at parent level
  const auto& parent_local_ref = parent->ref_by_into(ref_name);
  Refinement parent_reg_ref = *parent_local_ref;
  parent_reg_ref.location.devs[0].name = "REGISTER";
  parent_reg_ref.interior_shape = comp_reg_shape;
  parent->refs.erase(parent_local_ref);
  parent->refs.insert(parent_reg_ref);
  PropagateRefLoc(parent, parent_reg_ref);
}

static int count = 0;

// Replace the local refinements in place with register refinements
// in cache block and computing block
void ReplaceLocalRefinement(const AliasMap& parent_map,          //
                            Block* parent, Block* comp_parent,   //
                            Block* cache, Block* comp,           //
                            const std::string& ref_name,         //
                            const RegisterPassOptions& opt) {
  auto cache_inner = cache->SubBlock(0);
  auto comp_inner = comp->SubBlock(0);

  size_t load_size = comp_inner->exterior_shape(ref_name).sizes_product_bytes();
  if (load_size > opt.reg_size) {
    return;
  }

  if (count > 0) return;
  ++count;

  IVLOG(1, "Enter ReplaceLocalRefinement");
  IVLOG(1, *parent);

  const auto& cache_outer_local_ref = cache->ref_by_from(ref_name);
  const auto& cache_inner_local_ref = cache_inner->ref_by_from(cache_outer_local_ref->into());
  const auto& comp_outer_local_ref = comp->ref_by_into(ref_name);
  const auto& comp_inner_local_ref = comp_inner->ref_by_into(comp_outer_local_ref->into());

  // register refinement shape
  std::vector<Extent> comp_reg_extents = comp_inner_local_ref->Extents(comp_inner->idxs);
  std::vector<size_t> comp_reg_sizes;
  for (const auto& reg_ext : comp_reg_extents) {
    comp_reg_sizes.push_back(reg_ext.max);
  }
  auto comp_reg_shape = SimpleShape(comp_inner_local_ref->interior_shape.type,
                                    comp_reg_sizes, comp_inner_local_ref->interior_shape.layout);

  // New register refinements in cache block
  Refinement cache_inner_reg_ref = *cache_inner_local_ref;
  Refinement cache_outer_reg_ref = *cache_outer_local_ref;
  cache_inner_reg_ref.interior_shape = comp_reg_shape;
  for (auto& dim : cache_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  cache_inner_reg_ref.location.devs[0].name = "REGISTER";
  cache_inner_reg_ref.clear_tags();
  cache_inner->refs.erase(cache_inner_local_ref);
  cache_inner->refs.insert(cache_inner_reg_ref);
  size_t n_dim = comp_reg_shape.dims.size();
  for (size_t i = 0; i < n_dim; ++i) {
    cache_outer_reg_ref.interior_shape.dims[i].stride = comp_reg_shape.dims[i].stride;
  }
  cache_outer_reg_ref.location.devs[0].name = "REGISTER";
  cache_outer_reg_ref.clear_tags();
  ClearAccesses(cache_outer_reg_ref);
  cache->refs.erase(cache_outer_local_ref);
  cache->refs.insert(cache_outer_reg_ref);

  // New register refinements in computing block
  Refinement comp_inner_reg_ref = *comp_inner_local_ref;
  Refinement comp_outer_reg_ref = *comp_outer_local_ref;
  comp_inner_reg_ref.interior_shape = comp_reg_shape;
  for (auto& dim : comp_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  comp_inner_reg_ref.location.devs[0].name = "REGISTER";
  comp_inner->refs.erase(comp_inner_local_ref);
  comp_inner->refs.insert(comp_inner_reg_ref);
  comp_outer_reg_ref.interior_shape = comp_reg_shape;
  comp_outer_reg_ref.location.devs[0].name = "REGISTER";
  ClearAccesses(comp_outer_reg_ref);
  comp->refs.erase(comp_outer_local_ref);
  comp->refs.insert(comp_outer_reg_ref);
 
  // Add the register refinement at parent level
  const auto& parent_local_ref = parent->ref_by_into(ref_name);
  Refinement parent_reg_ref = *parent_local_ref;
  parent_reg_ref.location.devs[0].name = "REGISTER";
  parent_reg_ref.interior_shape = comp_reg_shape;
  parent->refs.erase(parent_local_ref);
  parent->refs.insert(parent_reg_ref);

  IVLOG(1, *parent);

  IVLOG(1, "Leave");
}

// Replace the block, loading from global memory and store into registers
bool CacheRefInRegister(const AliasMap& parent_map,          //
                        Block* parent, Block* comp_parent,   //
                        Block* cache, Block* comp,           //
                        const RegisterPassOptions& opt) {

  std::set<Refinement>::const_iterator cache_ref_it;
  // Determine the refinement in cache block
  if (opt.dir == RefDir::In) {
    cache_ref_it = cache->ref_by_into("dst");
  } else if (opt.dir == RefDir::Out) {
    cache_ref_it = cache->ref_by_into("src");
  } else {
    throw std::runtime_error("Invalid direction for caching into registers.");
  }
  std::string ref_name = cache_ref_it->from;

  auto cache_inner = cache->SubBlock(0);
  const auto& cache_outer_local_ref = cache->ref_by_from(ref_name);
  const auto& cache_inner_local_ref = cache_inner->ref_by_from(cache_outer_local_ref->into());
  const auto cache_outer_global_ref = opt.dir == RefDir::In ? cache->ref_ins()[0] : cache->ref_outs(true)[0];
  const auto cache_inner_global_ref = cache_inner->ref_by_from(cache_outer_global_ref->into());

  if (cache_outer_local_ref->access == cache_outer_global_ref->access &&
      cache_inner_local_ref->access == cache_inner_global_ref->access) {
    // If global access and local access are same in the cache block,
    // replace the cache block referencing the computing block.
    // The global access and register access in the new cache block are
    // same as the original local access in the computing block.
    ReferenceComputingBlock(parent_map, parent, comp_parent, cache, comp, ref_name, opt);
    return true;
  }

  if (cache_outer_local_ref->has_tag("same_access")) {
    // If the cache block's local access is same as that in the computing block,
    // replace local refinement with register refinement and keep global access
    ReplaceLocalRefinement(parent_map, parent, comp_parent, cache, comp, ref_name, opt);
    return true;
  }

  return false;
}

void CacheWholeRefInRegister(Block* parent, Block* cache, Block* comp,  //
                             const RegisterPassOptions& opt) {
  std::set<Refinement>::const_iterator cache_ref_it;
  // Determine the refinement in cache block
  if (opt.dir == RefDir::In) {
    cache_ref_it = cache->ref_by_into("dst");
  } else if (opt.dir == RefDir::Out) {
    cache_ref_it = cache->ref_by_into("src");
  } else {
    throw std::runtime_error("Invalid direction for caching into registers.");
  }

  // The candidate must be the ref marked as LOCAL
  if (cache_ref_it->location.devs[0].name != "LOCAL") {
    return;
  }

  // Get the inner and outer loop counts for the cache block
  size_t outer_loop;
  size_t inner_loop;
  OuterInnerLoopCount(cache, &outer_loop, &inner_loop);

  auto cache_inner = cache->SubBlock(0);
  size_t load_size = cache->exterior_shape(cache_ref_it->into()).byte_size();
  if (load_size > opt.reg_size) {
    return;
  }

  // Now we compute the load count of the refinement elements in the computational block
  size_t comp_load_count = 0;
  size_t iloop;
  size_t oloop;
  OuterInnerLoopCount(comp, &oloop, &iloop);

  // If the block is threaded, only count the inner loop
  if (comp->has_tag("gpu_thread")) {
    comp_load_count += iloop;
  } else {
    comp_load_count += (iloop * oloop);
  }

  double cost = inner_loop * outer_loop * (opt.lmem_lat + opt.reg_lat);
  double benefit = comp_load_count * (opt.lmem_lat - opt.reg_lat);

  if (benefit > cost) {
    // Add a block loading from local and caching into registers
    auto reg_cache = CloneBlock(*cache);
    reg_cache->remove_tag("gpu_thread");
    reg_cache->set_tag("reg_cache_whole");
    auto reg_cache_inner = reg_cache->SubBlock(0);
    reg_cache_inner->set_tag("reg_cache_whole");
    InsertAfterBlock(parent, cache, reg_cache);
    // Create a local refinement
    auto parent_ref_it = parent->ref_by_into(cache_ref_it->from);
    std::string local_ref = cache_ref_it->from + "_local";
    Refinement parent_local_ref = parent_ref_it->WithInto(local_ref);
    parent->refs.insert(parent_local_ref);
    // Rename the local refinement in the original cache block
    cache_ref_it->mut().from = local_ref;
    // Replace the source refinement from raw to local
    auto src_ref_it = reg_cache->ref_by_into("src");
    reg_cache->refs.erase(src_ref_it);
    Refinement new_src_ref = cache_ref_it->WithInto("src");
    reg_cache->refs.insert(new_src_ref);
    // Change the register refinement in the parent block
    parent_ref_it->mut().location.devs[0].name = "REGISTER";
    PropagateRefLoc(parent, *parent_ref_it);
  }
}

void BlocksForRegisterCache(const AliasMap& parent_map,
                            Block* parent, Block* cache,
                            const RegisterPassOptions& opt) {
  std::string ref_name;
  if (opt.dir == RefDir::In) {
    ref_name = cache->ref_by_into("dst")->from;
  } else if (opt.dir == RefDir::Out) {
    ref_name = cache->ref_by_into("src")->from;
  } else {
    throw std::runtime_error("Invalid direction for caching into registers.");
  }

  Block* comp = nullptr;
  Block* comp_parent;

  // Find the parent of the computing block with comp_parent_tag
  if (parent->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent;
  } else if (parent->SubBlock(0)->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent->SubBlock(0).get();
  } else {
    IVLOG(1, "Cannot find the computing block.");
    return;
  }

  // Find the computing block inside comp_parent
  for (auto stmt : comp_parent->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner && !inner->has_tag("cache")) {
      // Sometimes inner doesn't have ref, ignore it
      if (inner->ref_by_from(ref_name, false) != inner->refs.end()) {
        comp = inner.get();
        break;
      }
    }
  }

  if (parent != comp_parent) {
    // In this case we need to check if there is any use of the refinement
    // in parent, which may not be safe
    for (auto stmt : parent->stmts) {
      auto inner = Block::Downcast(stmt);
      if (inner && inner.get() != cache && !inner->has_tag(opt.comp_parent_tag)) {
        if (inner->ref_by_from(ref_name, false) != inner->refs.end()) {
          return;
        }
      }
    }
  }

  // Now we can start the caching if computing block and its parent exist
  if (comp) {
    bool ret = CacheRefInRegister(parent_map, parent, comp_parent, cache, comp, opt);
    // It is not safe to let cache store be in registers
    if (!ret && opt.dir == RefDir::In) {
      CacheWholeRefInRegister(parent, cache, comp, opt);
    }
  }
}

static void RegisterCacheRecurse(const AliasMap& parent_map,   //
                                 Block* parent, Block* block,  //
                                 const Tags& reqs,             //
                                 const RegisterPassOptions& opt) {
  if (block->has_tags(reqs)) {
    if (!block->has_tag("reg_cache_partial") && !block->has_tag("reg_cache_whole")) {
      BlocksForRegisterCache(parent_map, parent, block, opt);
    }
  } else {
    for (auto& stmt : block->stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        AliasMap alias_map(parent_map, block);
        RegisterCacheRecurse(alias_map, block, inner.get(), reqs, opt);
      }
    }
  }
}

void RegisterCachePass::Apply(CompilerState* state) const {
  RegisterPassOptions opt;
  auto reqs = FromProto(options_.reqs());
  opt.local_loc = stripe::FromProto(options_.local_loc());
  opt.reg_loc = stripe::FromProto(options_.register_loc());
  opt.reg_size = options_.register_size();
  opt.gmem_lat = options_.global_memory_latency();
  opt.lmem_lat = options_.local_memory_latency();
  opt.reg_lat = options_.register_latency();
  opt.dir = stripe::FromProto(static_cast<stripe::proto::Refinement::Dir>(options_.dir()));
  opt.comp_parent_tag = options_.comp_parent_tag();
  opt.cache_index_order = options_.index_order() == "cache";
  opt.align_size = options_.align_size();

  AliasMap base;
  RegisterCacheRecurse(base, nullptr, state->entry(), reqs, opt);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<RegisterCachePass, proto::RegisterCachePass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
