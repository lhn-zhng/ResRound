# NVFP4 Warpgroup Sparse Fusion Experiment

目标：从当前 producer-warp side work 路线切出来，专门实验真正修改 CTA 内 warpgroup 划分，让 sparse correction 尽量隐藏在 dense GEMM 生命周期里。

## Baseline Source

本目录从 `collected/nvfp4_direct_add_success` 复制最小可跑实现，并改成独立 Python/extension 名：

- Python module: `nvfp4_warpgroup_sparse_fusion`
- Extension prefix: `nvfp4_warpgroup_sparse_fusion`
- Main kernel source: `src/tma_direct_add.cu`
- Main benchmark: `benchmarks/probe_scheduler_quality_compact_merge.py`
- Dense-only 4-WG smoke: `benchmarks/probe_dense_4wg.py`

## Modes

- `tail`: 旧 producer-warp 插活路径，`mode 6/7`，不改变 dense warpgroup 数量。
- `idle_stage`: 旧 ScaleA/ScaleB idle-stage 插活路径，`mode 8`。
- `extra_wg`: 新实验路径，`mode 9`。CTA 从 3 个 warpgroup 扩到 4 个 warpgroup，新增 `wg3` 做当前 dense tile 的 k-major bounded sparse work。
- `extra_wg_idle`: 诊断路径，`mode 10`。CTA 同样扩到 4 个 warpgroup，但 `wg3` 不做 sparse，只测试 512-thread CTA 和 dense 主路径同步是否兼容。
- `extra_wg_noprobe`: 最终 producer-only 诊断收敛路径，`mode 11`。仍由 `wg3` 做 sparse A/B load + CUDA-core FMA，但去掉 checksum/reduction/sink 写回，只保留 inline asm use 防止编译器删掉计算。
- `extra_wg_write_noprobe`: sparse 写入诊断路径，`mode 12`。在 `mode 11` 基础上让 `wg3` 把 compact per-entry delta 写到独立 `delta_entries`，不做 dense output merge，也不写 probe sink。
- `extra_wg_add_noprobe`: final add 诊断路径，`mode 13`。`wg3` 写 compact delta 后等待 dense tile 完成，再用 per-entry BF16 atomic add 回 dense output。结果极慢，只作为负例。
- `extra_wg_rowadd_noprobe`: final add 诊断路径，`mode 14`。不用 atomic，但按 row 扫 k-major entries 做 late add。因为重复扫描 metadata，结果更慢，只作为负例。
- `extra_wg_metaadd_noprobe`: 当前最好的 in-kernel final add 诊断路径，`mode 15`。复用 merge metadata，让 `wg3` 等 dense tile 完成后做 row-local reduce/add，不再单独 launch merge kernel。
- `extra_wg_smem_merge_noprobe`: epilogue shared-memory merge 诊断路径，`mode 18`。`wg3` 仍先写 compact `delta_entries`，dense epilogue 在 TMA store 前把 delta 加进 swizzled `sC`，不再 launch external merge。功能正确，但 0.1% 下 `linear_proj`/`linear_fc2` 都慢于 r014 compact write + external merge，详见 `results/extrawg_r021_smem_merge_report.md`。
- `extra_wg_sharedacc_smem_noprobe`: CTA-local shared accumulator 诊断路径，`mode 19`。`wg3` 把 sparse delta 写进 shared memory `float sparse_acc[AccRows, CtaN]`，dense epilogue 在 TMA store 前合入 `sC`。`AccRows=48` 能跑但不 exact，full-row exact 受 shared memory 容量限制，详见 `results/extrawg_r032_sharedacc_smem_report.md`。
- `extra_wg_subacc32_smem_noprobe`: staged CTA-local shared accumulator 诊断路径，`mode 20`。保持 dense `EpiN=64`，但 sparse accumulator 内部按 `SubN=32` 分 4 stage 合入 `sC`。功能 exact，但 0.1% 下明显慢于 r014，详见 `results/r033_epin_staged_sharedacc_design.md`。
- `dense_4wg`: 新 dense-only 入口。CTA 从 3 个 warpgroup 扩到 4 个 warpgroup，`wg0/wg1/wg2` 跑 dense，`wg3` 作为 sparse 预留角色通过入口同步后返回。

## Current Finding

`extra_wg` 不是简单把 `LaunchThreads` 从 `3*128` 改成 `4*128` 就能跑。当前 v13 dense kernel 对 CTA 内同步和 warpgroup register allocation 有隐含假设：

- `tail` 单 case 能返回，说明复制后的旧路径可用。
- `extra_wg` 单 case 卡住。
- `extra_wg_idle` 单 case也卡住，即使 `wg3` 不做 sparse。
- 尝试让 `wg3` 参与 epilogue consumer barrier，并把 consumer barrier participant 从 `256` 改为 `384` 后仍卡住。
- `dense_4wg` r006 已能正确返回，但它暂时禁用了 force 4-WG 路径里的 `setmaxnreg`，只作为诊断。
- `dense_4wg` r007 已恢复 4-WG 专用 `setmaxnreg`：`wg0/wg3` 先降到 low-reg，512-thread CTA barrier 确认 low-reg 角色到位，`wg3` 返回，`wg1/wg2` 再提升到 176 regs。r007 输出与原 TMA dense 完全一致。详见 `results/dense4wg_r007_reg176.md`。
- `extra_wg` r008 已把 `wg3` sparse CUDA-core load+FMA 接入 r007 协议，并且不再让 `wg3` 参与 dense epilogue barrier。`linear_proj` Load+FMA/TMA=1.132，`linear_fc2` Load+FMA/TMA=1.043，dense output max_abs=0。详见 `results/extrawg_r008_probe.md`。
- consumer regs 向上试探：184 能编译但 dense-only 运行挂；180 不能编译，因为 `setmaxnreg` 参数必须是 8 的倍数。因此当前继续使用 176。
- `extra_wg` r011 支持在保留 4-WG CTA 的同时只启用 `wg3` 内前 N 个 sparse warp。当前建议：`linear_fc2` 用 `side_warps=1`，`linear_proj` 用 `side_warps=4`。
- `extra_wg_noprobe` r013 去掉 r008/r012 的 checksum/reduction/sink 诊断成本后，`linear_fc2` Load+FMA/TMA=1.044，`linear_proj` Load+FMA/TMA=1.132。相对 r012 只分别省约 0.004 ms 和 0.005 ms，说明诊断 sink 不是主瓶颈。进一步用 `extra_wg_idle` 对照发现，4-WG 空壳本身已经接近 r013 时间：`linear_fc2` idle=0.8645 ms vs noprobe=0.8641 ms，`linear_proj` idle=0.3122 ms vs noprobe=0.3168 ms。当前大头是 4-WG 调度/同步/寄存器协议的固定成本，真实 sparse load/FMA 只占小头。详见 `results/extrawg_r013_fc2_noprobe.md`、`results/extrawg_r013_proj_noprobe.md` 和 `results/extrawg_r013_idlecheck.md`。
- `extra_wg_write_noprobe` r014 已把 sparse compact delta 写入接进同一个 4-WG fused kernel。`linear_fc2` dense+sparse-write/TMA=1.058，`linear_proj` dense+sparse-write/TMA=1.176；相对 no-write r013，写入额外增加约 0.012 ms 和 0.015 ms。dense output max_abs 仍为 0，merge 后 delta abs sum 非零，说明 sparse 写入确实发生但没有污染 dense output。详见 `results/extrawg_r014_fc2_write_noprobe.md` 和 `results/extrawg_r014_proj_write_noprobe.md`。
- final add 尝试后，当前最好的单 kernel 版本是 `extra_wg_metaadd_noprobe` r018：`linear_fc2` 1.0193 ms，`linear_proj` 0.3975 ms。它正确完成 dense+sparse+add，但仍慢于 r014 写出 + 独立 merge：`linear_fc2` 0.9458 ms，`linear_proj` 0.3584 ms。结论是 final add 不适合塞进当前 dense CTA 的 `wg3` tail；独立 active-row merge kernel 的并行度更合适。详见 `results/extrawg_r018_fc2_metaadd_barrier.md` 和 `results/extrawg_r018_proj_metaadd_barrier.md`。
- r021 尝试把 merge 提前到 dense epilogue shared-memory tile：`linear_proj` 0.1% 为 0.4653 ms，`linear_fc2` 0.1% 为 1.0856 ms，delta abs sum 与 r014 external merge 对齐。但它必须让 dense epilogue 等 `wg3` compact write 完成，并在 TMA store 前扫 row metadata，反而比 r014 更慢。结论是不要把 compact delta merge 塞进当前 epilogue；后续应继续压 external merge 或改变 delta 表达。详见 `results/extrawg_r021_smem_merge_report.md`。
- r022 用 NCU 重新诊断 external active-row merge：`linear_fc2` 1% old merge 为 284 us、DRAM throughput 92.5%、DRAM read 341 MB，确认瓶颈是 compact `delta_entries[entry, N]` + output 的全局内存流量。`vec8` 修好 sector 利用率但不降低 DRAM bytes，`fastpath` 也基本无效。新候选 `TMA dense + direct_sparse_add` 不再写/读 compact delta，而是直接从 selected row payload 和 BF16 `B[k, :]` 重算 correction 并加回 output。实际串联计时：`linear_fc2` 1% 从 r014 `1.1837 ms` 降到 `0.9824 ms`，`linear_proj` 1% 从 `0.7824 ms` 降到 `0.3683 ms`；对应单独 standalone sparse kernel 分别为 `0.0812 ms` 和 `0.0627 ms`。它仍是两 kernel 路径，不是完全隐藏的单 kernel，但已经是当前最有价值的 exact merge 替代方向。详见 `profile/merge_external_active_row_r022_20260624/REPORT.md`。
- r023 继续压 `direct_sparse_add` 的局部成本，排除了两个自然优化：packed output read 把 `linear_fc2` 1% global LD sectors 从 `44.98M` 降到 `22.88M`，但 DRAM read 仍是 `118.4 MB`，duration 从 `129.6 us` 变成 `131.5 us`；vec16 direct-add 正确但更慢，`linear_fc2` 1% 为 `0.1028 ms` vs vec8 `0.0876 ms`，`linear_proj` 1% 为 `0.0782 ms` vs `0.0646 ms`。结论是 direct-add 剩余成本不在 output scalar load 或 vec 宽度，而在 B/output 的 DRAM/L2 长延迟、写回流量和 row work 分布。详见 `profile/direct_sparse_add_packread_r023_20260624/REPORT.md`。
- r024 验证 row work 分布确实影响 direct-add。只重排 `active_rows`，不改变 outlier 选择和 correction 数值：sparse-only direct-add 在 `linear_fc2` 1% 从 `0.1556 ms` 降到 `0.1188 ms`，`linear_proj` 1% 从 `0.0901 ms` 降到 `0.0778 ms`。真实 `TMA dense; direct_sparse_add` pipeline 的 1% 标准 2/5 确认：`linear_fc2` 从 `0.9875 ms` 降到 `0.9461 ms`，`linear_proj` 从 `0.3698 ms` 降到 `0.3572 ms`。新增 `reorder_active_rows_by_nnz(..., mode="heavy_light")`，当前 exact two-kernel baseline 应使用 `TMA dense + direct_sparse_add + active_rows heavy/light scheduling`。详见 `profile/direct_sparse_add_row_schedule_r024_20260624/REPORT.md`。
- r025 把 r024 的 `heavy_light` scheduling 固化成标准 `2/5` 全 ratio baseline，并补测 standalone sparse kernel。`linear_fc2` 1% 的 `TMA dense; direct_sparse_add` 从 row-id `0.9857 ms` 降到 `0.9482 ms`，standalone sparse kernel 为 `0.0848 ms`，相对 r014 `1.1837 ms` 快 `19.9%`，但仍比 CUTLASS dense 慢 `18.2%`；`linear_proj` 1% 从 row-id `0.3666 ms` 降到 `0.3557 ms`，standalone sparse kernel 为 `0.0762 ms`，相对 r014 `0.7824 ms` 快 `54.5%`，但仍比 CUTLASS dense 慢 `8.2%`。当前 exact two-kernel baseline 正式更新为 `TMA dense + direct_sparse_add + active_rows heavy_light scheduling`。详见 `profile/direct_sparse_add_scheduled_r025_20260624/REPORT.md`。
- r026 尝试把 direct-add kernel 从 linear grid 改成 rowblock grid，避免 CTA 跨 row。结果正确但基本中性：`linear_fc2` 全比例 rowblock 都略慢或持平，`linear_proj` 0.1%/0.2%/1% 只快不到 `0.5%`。因此 CTA 跨 row 不是主要瓶颈，r025 的 linear-grid direct-add + heavy_light scheduling 仍是当前 exact two-kernel baseline。详见 `profile/direct_sparse_add_rowblock_r026_20260624/REPORT.md`。
- r027 补上 standalone sparse kernel 时间并用 NCU 诊断 direct-add。1% 下 standalone sparse kernel 为 `linear_fc2 0.0848 ms`、`linear_proj 0.0762 ms`；NCU 对比显示 standalone 便宜是因为它不读 dense output，而 direct-add 的额外成本主要来自 output/B global load 和 long-scoreboard stall。新增 128-bit `vstore` 可把 store sector 利用率从 `16 B/sector` 修到 `32 B/sector`，但 `linear_fc2` 1% direct-add 仍是 `125.2 us` 量级，因为 DRAM read 没降。因此 r025 仍是当前 exact two-kernel baseline，真正零成本 merge 需要在 dense output 首次 global store 前合入 sparse delta。详见 `profile/direct_sparse_add_scheduled_ncu_r027_20260624/REPORT.md`。
- r028 试了更接近真融合的 `pre-store shared-memory direct correction`：dense epilogue 先把 accumulator 写入 shared-memory `sC`，再直接计算 sparse correction 加进 `sC`，最后 TMA store。它避免了 global output read，但结果是负例：`linear_fc2` 1% 从 post-store direct `0.9988 ms` 变成 `1.1820 ms`，`linear_proj` 1% 从 `0.4463 ms` 变成 `0.6574 ms`。barrier-only 基本无成本，瓶颈是 dense epilogue 被 active-row sparse metadata scan/B load/FMA 阻塞，TMA store 必须等 sparse loop 完成。因此 pre-store merge 只有在 delta 已经是 fragment-local、几乎不扫 metadata 时才有希望。详见 `profile/direct_smem_correction_r028_20260624/REPORT.md`。
- r029 专门 profile 0.1% 低 ratio 下的 direct-add vs standalone sparse kernel。`linear_fc2` direct-add NCU duration `57.6 us` vs standalone `41.0 us`，`linear_proj` `15.5 us` vs `12.9 us`。关键证据是 direct-add 多出来的 DRAM read 精确等于 active rows 的完整 dense output 读取量：`linear_fc2` 约 `61.69 MB`，`linear_proj` 约 `7.36 MB`。这证明低 ratio 下也不是 metadata 或 sparse FMA 主导，而是 post-store merge 必须读 dense output 的物理成本。结论：继续微调 direct-add 只能小幅改善；要接近零成本，必须让 side sparse producer 在 dense global store 之前准备好 tile-local/fragment-local delta，epilogue 只做便宜 add。详见 `profile/direct_sparse_add_lowratio_ncu_r029_20260624/REPORT.md`。
- r030 做了 `precomputed-delta shared-memory add` 下限实验：先用 standalone sparse kernel 生成完整 BF16 `delta_output[M,N]`，dense epilogue 不扫 metadata、不 load `B`、不做 sparse FMA，只在 TMA store 前把 `delta_output[row,n_tile]` 加进 `sC`。结果正确但仍是负例：`linear_fc2` 0.1%/1% 分别比 dense 慢 `+20.86%/+38.02%`，`linear_proj` 0.1%/1% 分别慢 `+19.76%/+80.49%`。NCU 在 `linear_proj` 0.1% 上显示 precomputed 比 dense 多读 `7.369 MB`，正好等于 `898 active_rows * 4096 * 2B` 的 full-row delta 读取量。结论：global full-row delta 即使预先算好，也会在 epilogue 临界路径形成新的 `active_rows*N*2B` 读流量；继续追完全隐藏必须让 delta 在 CTA-local/fragment-local 路径里准备好，不能经过 global `[M,N]` delta buffer。详见 `profile/direct_smem_precomputed_delta_r030_20260624/REPORT.md`。
- r031 复测 `per-row atomic delta write` 作为减少后置 merge delta 读取的候选。它把 compact `delta_entries[entry,N]` 改成 global BF16 atomic 聚合的 dense `delta_output[row,N]`，理论上 merge 每个 active row 只读一条 delta row；但同 r014 0.1% selector 下，producer 写入已经慢到 `linear_proj 0.9968 ms`、`linear_fc2 3.3300 ms`，远高于 r014 compact write + external merge pipeline。结论：global atomic row-delta 写入不可用；如果要减少 delta 读取条数，必须走 CTA-local/fragment-local 聚合，而不是 global atomic。详见 `results/merge_atomic_rowdelta_r031_report.md`。
- r032 尝试 `CTA-local shared accumulator merge`：`wg3` 把 sparse delta 写入 shared memory `float sparse_acc[AccRows, CtaN]`，dense epilogue 在 TMA store 前合入 `sC`，避免 global delta round-trip。`AccRows=48` 能跑但只覆盖约 `81%` r014 delta，且仍比 r014 exact 更慢：`linear_proj` 0.3883 ms vs r014 0.3618 ms，`linear_fc2` 1.0190 ms vs r014 0.9718 ms；`AccRows=64` 在 dense TMA launch 阶段即 `CUDA error: invalid argument`，说明 shared memory 上限阻断 full-row exact。结论：不要继续扩大 full-tile float sharedacc，下一步只能考虑 `EpiN` 粒度/fragment-local delta。详见 `results/extrawg_r032_sharedacc_smem_report.md`。
- r033 继续测试 staged CTA-local shared accumulator。`HANDWRITTEN_TMA_EPIN=32` dense gate 失败，当前 TMA store/`sC` layout 会产生 NaN；保留默认 `EpiN=64`、只把 sparse shared accumulator 切成 `SubN=32` 后，结果 exact 但非常慢：0.1% 下 `linear_proj` 0.9446 ms、`linear_fc2` 2.7712 ms，远慢于 r014 exact pipeline。结论：如果 fragment/tile-local merge 需要 dense epilogue 等 staged sparse producer，它也不成立。详见 `results/r033_epin_staged_sharedacc_design.md`。
- r034 复查 single-kernel post-store direct-active 的 block-local row schedule：去掉 padding 并测试 `row_id`/`nnz_desc`/`nnz_asc`/`heavy_light` 后仍慢于 r025 two-kernel direct-add。最佳点对比 r025：`linear_fc2` 0.1% `0.9238 ms` vs `0.8907 ms`，1% `0.9917 ms` vs `0.9482 ms`；`linear_proj` 0.1% `0.3330 ms` vs `0.2982 ms`，1% `0.4252 ms` vs `0.3557 ms`。结论：post-store 单 kernel tail 的问题不是 padding，而是并行度低和仍需 reread dense output。详见 `profile/direct_active_block_schedule_r034_20260624/REPORT.md`。
- r035 测试 exact direct-add 的 cache-policy 局部优化：新增 `sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore`，对 `B[k, :]` load 加 `ld.global.L1::evict_last` 并使用 128-bit store。`linear_fc2` 1% NCU 显示 DRAM read 仍是 `118.383 MB`，global load bytes/sector 仍是 `7.70`，global LD inst 仍是 `4.276M`；主要变化只是 store bytes/sector 从 `16` 到 `32`。结论：收益来自已有 vstore，不是 B cache hint，不能解决 post-store merge 下界。详见 `profile/direct_add_cache_policy_r035_20260624/REPORT.md`。
- r036 测试 int16-column metadata direct-add：新增 `sparse_active_row_col_value_payload_vec8_inplace_vstore`，用 `row_payload.row_ks` 避免 `flat - row*K`，并结合 packed output load + 128-bit store。结果全点变慢，`linear_fc2` 1% 的 `10/50` 复跑为 `flat_vstore=0.0889 ms`、`col_vec8=0.0978 ms`。NCU 显示它把 global load bytes/sector 从 `7.70` 提到 `15.10`，但 DRAM read 不降，L1 hit 从 `87.50%` 降到 `78.19%`，long scoreboard 从 `29.81` 增到 `32.96`。结论：metadata 算术不是主瓶颈，post-store output reread 下界仍在。详见 `profile/direct_add_col_payload_r036_20260624/REPORT.md`。
- r037 测试 4-WG tailassist direct-active：force 4-WG dense CTA，并让 `wg3` 在 dense store 后参与 post-store direct-add，把 active row work 从 8 个 consumer warp 分到 12 个 warp。`10/30` focused 结果显示 `linear_fc2` 基本无收益或变慢，`linear_proj` 1% 只从 `0.4218 ms` 降到 `0.4156 ms`，仍慢于 r025 two-kernel direct-add `0.3557 ms`。NCU 显示 4WG 确实把 eligible warps/cycle 从 `0.386` 提到 `0.499`、long-scoreboard 从 `1.565` 降到 `1.260`，但 DRAM read/write 和 global sectors 基本不变。结论：WG3 参与只能改善少量 latency hiding，不能消掉 post-store output reread 下界。详见 `profile/direct_active_4wg_tailassist_r037_20260624/REPORT.md`。
- r038 测试 exact `light-row pre-store + heavy-row post-store` hybrid：把 `row_nnz <= threshold` 的轻行放进 dense TMA pre-store smem correction，重行继续 post-store direct-add，并分别测试 `nnz_desc`/`heavy_light` 重行顺序。所有测试点最优 threshold 都是 `0`，即不启用 pre-store；每个点里最快的 threshold>0 仍比 threshold=0 慢 `+14%` 到 `+27%`。结论：当前 pre-store smem 分流仍会把 sparse work 放进 dense critical path，不是可用 merge 策略。详见 `profile/direct_smem_lightrow_hybrid_r038_20260624/REPORT.md`。
- r039 测试 exact tile-col regional post-store correction：dense TMA store 后按 `(128-row M tile, 128-col N tile)` 区域做 sparse add。最佳 tile-col 仍比 active-row direct-add 慢：`linear_fc2` 慢 `+3.1%/+4.5%`，`linear_proj` 慢 `+5.7%/+5.1%`。persistent ready-queue smoke 在 `linear_proj 0.1%` 慢 `+245.8%`。结论：按区域扫描 rowblock 会浪费空行 work，当前分区域 post-store merge 不如 active-row direct-add。详见 `profile/tile_col_poststore_r039_20260624/REPORT.md`。
- r040 测试 exact direct-add 的 single/double-row fastpath：对 `row_nnz==1/2` 展开 sparse loop，并和已有 128-bit `vstore` direct-add 对照。`10/50` focused 复测显示 fastpath 相对 vstore 没有稳定 pipeline 收益：`linear_fc2` 0.1%/1% 分别慢 `+1.28%/+1.35%`，`linear_proj` 0.2%/1% 分别慢 `+0.65%/+0.34%`。结论：row loop/metadata 不是主瓶颈，继续微调 post-store direct-add 指令不值得；瓶颈仍是 output reread/writeback。详见 `profile/direct_add_fastpath_r040_20260624/REPORT.md`。
- r052-r058 复查 full-topk 0.1% 的 `linear_proj` threshold/schedule 和 EpiN-local delta 可行性：threshold=68 的快点不能稳定复现，当前仍应保留 `dense_light_direct_hot` threshold=64 + `vstore`；row order 差异只有约 `0.00045 ms`，不是主要优化方向。EpiN=64 的 local FP32 delta footprint 上界只有 `32 KiB`，但 post-store output RMW 下界在 `linear_fc2` 已达 `255.92 MiB`，说明下一步应转向真正 EpiN/tile-local delta，在 dense 首次 store 前合入，而不是继续压 post-store direct-add。详见 `results/full_topk_threshold_schedule_epin_r052_r058_findings.md`。
- r059-r060 把 standalone k-major delta producer 从固定 `EpiN=64` 模板化为 `EpiN=32/64`，并在 full-topk 0.1% 上实测。r060 正确性通过：`linear_proj` max_abs `0.000977`，`linear_fc2` max_abs `0.000244`；但 global delta-store 仍为毫秒级，`linear_proj` `1.21-1.28 ms`、`linear_fc2` `1.66-1.70 ms`。结论：该路径只作为 producer 成本/正确性 probe，不是候选 fused path；下一步应优先做 in-CTA `EpiN=64` local-delta，`EpiN=32` 作为共享内存/寄存器压力 fallback。详见 `results/full_topk_kmajor_epin_delta_r059_r060_findings.md`。
- r061 复跑 full-topk hotblock+dense hybrid：`linear_proj` 0.1% 最好仍是 `skip8 + HANDWRITTEN_TMA_STAGES=2 + LOCAL_DELTA_STAGE_BUFFERS=2`，build-included hybrid 为 `0.453344 ms`，max_abs `0.001953`，比 poststore 快 `7.65%` 但仍比 dense 多 `0.162 ms`。`linear_fc2` full local-delta 即使用 side_warps=4 也为 `1.100320 ms`，比 poststore 慢 `10.23%`，因此 fc2 不应走当前 local-delta 主线。详见 `results/full_topk_local_delta_hotblock_r061_findings.md`。
- r062/r001 在 r061 基础上增加 host-side per-128-row M-block light-entry budget：只把预算内的 light rows 喂给 WG3 local-delta，预算外 light rows 和 heavy rows 一起走 dense residual fallback，因此仍保持 full-topk exact。`linear_proj` 0.1% focused `3/10` 下，budget 48/64/96 都稳定优于 r061-style no-budget，最佳 budget 96 的 build-included hybrid 为 `0.404448 ms`，比同轮 poststore `0.477632 ms` 快 `15.32%`，比 dense 多 `0.122464 ms`，max_abs `0.001953`。当前有用区间是 block budget `48-96`；budget 16 会把太多轻行推入 residual，反而变慢。详见 `results/full_topk_hotblock_bounded_r001_findings.md`。
- r063 设计 EpiN-local pre-store delta 的真正下一版：现有 r061/r062 仍复用 row-wise local-delta producer，dense epilogue 容易等 sparse；新路径应把 full-topk payload 拆成 disjoint `local_payload` + `fallback_payload`，local 部分按 128-row M block 做 whole-row bounded admission，并以 k-major payload 喂 WG3，直接写 `SparseLocalDeltaStorage.tile[stage][local_row,EpiN]`，fallback 继续 exact direct-add。0.1% 数据上建议先测 `linear_proj` r063a `(row_nnz_cap=2,budget=64)`：local 465 rows / 601 entries / 468 unique-K，避免 `7.27 MiB` post-store RMW，只需约 `3.66 MiB` k-major B；再测 r063b `(4,64)`：避免 `8.72 MiB` RMW、约 `5.15 MiB` k-major B。`linear_fc2` 先只做保守诊断 `(2,32)` 或 `(4,16)`，更大 budget 会引入 `28 MiB+` 到 `100 MiB+` local k-major B，预计难隐藏。详见 `results/full_topk_epin_local_prestore_design_r063.md`。
- r063 实装 smoke：新增 `extra_wg_kmajor_local_delta_db_smem_noprobe` / mixed CTA `32`，host 侧用 `build_kmajor_payload_from_active_rows` 保证 local k-major payload 和 admitted rows 同源，fallback disjoint。正确性通过，所有点 `max_abs=0.001953`；但当前 BF16 shared-memory read-modify-write k-major producer 不赢 r062：`linear_proj` 0.1% r063a `(cap=2,budget=64)` build-included `0.464576 ms`，r063b `(4,64)` `0.474240 ms`，r063c `(4,96)` `0.521024 ms`，row-owner 4-warp r063b `0.492512 ms`；同轮 r062 row-wise recheck `(threshold=8,budget=96)` 为 `0.406208 ms`。结论：这个 k-major local-delta backend 正确但不适合作为性能主线；下一版需要 admitted-row/fragment-local FP32 accumulator，最后一次性写 BF16 local_delta，而不是 per-entry BF16 shared RMW。详见 `results/full_topk_epin_local_prestore_r063_findings.md`。

因此当前 blocker 已收窄：512-thread CTA 本身可以合法返回；原 3-WG 的 `setmaxnreg` 寄存器重分配模型不能直接搬到 4-WG force dense-only 路径，但 r007 的 4-WG low/high-reg 协议可作为后续接 sparse 的基础。

## Repro Commands

```bash
NVFP4_DIRECT_ADD_TMA_VARIANT=wg_repartition_r001 \
conda run -n transformer_engine --no-capture-output \
python collected/nvfp4_warpgroup_sparse_fusion/benchmarks/probe_scheduler_quality_compact_merge.py \
  --module-suffixes linear_proj \
  --score-modes sum_sq \
  --group-budget linear_proj:19 \
  --side-warps 1 \
  --side-modes tail \
  --warmup 1 --iters 2 --skip-quality --merge-mode old
```

```bash
NVFP4_DIRECT_ADD_TMA_VARIANT=wg_repartition_r007 \
conda run -n transformer_engine --no-capture-output \
python collected/nvfp4_warpgroup_sparse_fusion/benchmarks/probe_scheduler_quality_compact_merge.py \
  --module-suffixes linear_proj \
  --score-modes sum_sq \
  --group-budget linear_proj:19 \
  --side-modes extra_wg_idle \
  --warmup 1 --iters 2 --skip-quality --merge-mode old
```

```bash
NVFP4_DIRECT_ADD_TMA_VARIANT=dense4wg_r006 \
conda run -n transformer_engine --no-capture-output \
python collected/nvfp4_warpgroup_sparse_fusion/benchmarks/probe_dense_4wg.py \
  --module-suffixes linear_proj linear_fc2 \
  --warmup 2 --iters 5 \
  --out-json collected/nvfp4_warpgroup_sparse_fusion/results/dense4wg_r006.json \
  --out-md collected/nvfp4_warpgroup_sparse_fusion/results/dense4wg_r006.md
```

```bash
NVFP4_DIRECT_ADD_TMA_VARIANT=dense4wg_r007_reg176 \
conda run -n transformer_engine --no-capture-output \
python collected/nvfp4_warpgroup_sparse_fusion/benchmarks/probe_dense_4wg.py \
  --module-suffixes linear_proj linear_fc2 \
  --warmup 2 --iters 5 \
  --out-json collected/nvfp4_warpgroup_sparse_fusion/results/dense4wg_r007_reg176.json \
  --out-md collected/nvfp4_warpgroup_sparse_fusion/results/dense4wg_r007_reg176.md
```

## Next Design Direction

下一步不应继续在 old producer-warp side work 上加补丁。当前推荐顺序：

1. 从 `extra_wg` r011 开始使用 layer-specific sparse warp 粒度：`linear_fc2 side_warps=1`，`linear_proj side_warps=4`。
2. 保持 `wg3` 低寄存器 CUDA-core role，先不要把 consumer reg count 拉回 184/192/208；184 已经运行挂。
3. producer-only sparse load+FMA 稳定后，不应再以 compact delta merge 为主线；r022 已证明 compact delta 表达本身导致后置 merge 读写过大。
4. exact 两 kernel baseline 应切到 `TMA dense + direct_sparse_add + active_rows heavy/light scheduling`，继续用它和 r014 做端到端正确性/性能对照。
5. r033 后不要继续做“dense epilogue 等 sparse stage”的 CTA-local merge。它虽然能消掉 global delta round-trip，但同步和重复 B load/FMA 会把 dense store 临界路径拖慢。
6. 如果继续追单 kernel 完全隐藏，需要设计 dense 结束后的第二阶段，让 CTA 变成 sparse row/N-chunk workers，并继承 r024 的 balanced active-row schedule；单个 dense CTA 内的 `wg3` 没有 standalone direct add 的并行度。
7. r013 已证明 checksum/sink 诊断成本不是 producer-only 路径的主要开销；`extra_wg_idle` 又证明 fixed 4-WG overhead 占了 producer-only overhead 的大部分。继续优化应优先降低“新增 warpgroup 改变 dense 主路径”的固定代价，例如减少 4-WG barrier/setmaxnreg 影响、避免整 CTA 扩到 512 threads，或回到已有 warpgroup 内寻找真实 idle slots，而不是继续微调 probe sink。

最终目标仍是 `linear_proj` 和 `linear_fc2` 真实 0.1% sparse+dense fused 不超过 CUTLASS dense 的 1.2x。
