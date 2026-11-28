"""Render the completed procedural pilot's frozen-actor evidence."""
import json
import os
from pathlib import Path

from .gcp_evaluation import TASKS


def fmt(value):
    return "无样本" if value is None else f"{value:.2f}"


def ratio(data):
    return f'{data["count"]}/{data["n"]}'


def write_report(run, best, final, out):
    run = Path(run)
    def link(label, path):
        return f"[{label}]({os.path.relpath(path, run)})"
    def sources(row):
        return " · ".join((link("权重", row["checkpoint"]), link("分母与分层指标", row["result_file"]),
                           link("逐起点、逐分支与逐帧记录", row["trajectories"])))
    summary = best["summary"]
    findings = []
    for task, data in summary["task_funnel"].items():
        score = data["unconditional_safe_success"]
        if not score["count"]:
            findings.append(f"{task} 的自然起点安全成功为 {ratio(score)}")
    for panel, data in summary["same_snapshot"].items():
        if not data["both_goals_met"]["count"]:
            findings.append(f"{panel} 同快照双目标成功为 {ratio(data['both_goals_met'])}")
    for panel, data in summary["slip_recovery"].items():
        for mode in ("fresh", "prepared"):
            group = data[mode]
            if not group["actual_losses"] or not group["safe_completed_after_loss"]:
                findings.append(f"{panel}/{mode} 实际失握 {group['actual_losses']} 次，安全恢复 {group['safe_completed_after_loss']} 次")
    for task in TASKS:
        data = summary["tracking"]["all_actor_trajectories"][task]
        score = data["strata"]["all"]
        bound = 12. if task in ("place", "pick", "planar_push") else 5.
        if score["p95"] is None or score["p95"] > bound or (bound == 12. and score["max"] > 30.):
            findings.append(f"{task} 自主轨迹跟踪 P95 / Max 为 {fmt(score['p95'])} / {fmt(score['max'])} {data['unit']}")
    lines = ["# 五类短训因果与跟踪核验报告", "",
             ("**建议暂缓 36 小时正式长训。** 选中权重的实测薄弱项如下。" if findings else
              "**建议放行 36 小时训练。** 选中权重在六类自然起点、各组配对指令和双向失握中均出现了有效完成，跟踪误差处于此前开发参考范围。"),
             ""]
    lines.extend("- " + finding for finding in findings)
    lines += ["", "本次评估使用冻结策略、程序化开发种子和单侧物理指标。"
              "建议依据短训是否呈现可复现的任务、指令和真实失握恢复信号，并参考历史跟踪上限。"
              "这些是本次投入决策的依据。项目尚未规定短训各任务最小成功率，正式 Step 1B 的未见仿真分布与基线比较须另外验收。",
              "", f"每个自然任务及每个因果场景预抽 {best['n_per_scenario']} 个起点。种子起始值为 {best['first_seed']}。"
              "未入场计入无条件分母，同快照比较保持物理状态、伺服状态、随机流和非目标策略输入一致。"
              "所有准备段和干预段均由冻结 Actor 执行。", "",
              f"- 选中 SHA256 `{best['sha256']}`。{sources(best)}",
              f"- 最终 SHA256 `{final['sha256']}`。{sources(final)}",
              f"- {link('全存档选模清单', Path(out) / 'selection.json')}", ""]
    snapshots = [("选中权重", best)]
    if final["sha256"] != best["sha256"]:
        snapshots.append(("最终权重", final))
    else:
        lines += ["最终权重与选中权重字节相同，以下结果适用于两者。", ""]
    for name, row in snapshots:
        summary = row["summary"]
        lines += [f"## {name}", "", "### 六类自然起点", "",
                  "|任务|入场 / 起点|成功 / 入场|安全成功 / 起点（95% Wilson 区间）|穿模 >1 mm / 起点|穿模 P95 mm|",
                  "|---|---:|---:|---:|---:|---:|"]
        for task, f in summary["task_funnel"].items():
            ci = f["unconditional_safe_success"]["ci95"]
            lines.append(f"|{task}|{ratio(f['entry_coverage'])}|{ratio(f['conditional_success'])}|"
                         f"{ratio(f['unconditional_safe_success'])} ({ci[0]:.0%}–{ci[1]:.0%})|"
                         f"{ratio(f['penetration_over_1mm'])}|{fmt(f['penetration_p95_mm'])}|")
        lines += ["", "安全成功要求末尾连续 10 步满足目标，整局禁止穿模最大值严格小于 1 mm。", "",
                  "### 同快照指令检验", "",
                  "|指令对|共同入场 / 起点|双目标成功 / 起点|实测终点分离中位数|指令分离|",
                  "|---|---:|---:|---:|---:|"]
        for panel, data in summary["same_snapshot"].items():
            lines.append(f"|{panel}|{data['both_entered']}/{data['starts']}|{ratio(data['both_goals_met'])}|"
                         f"{fmt(data['endpoint_separation_median'])} {data['unit']}|{data['expected_separation']} {data['unit']}|")
        lines += ["", "### 15° 自主滑脱重抓", "",
                  "|开门方向 / 开始条件|可用 / 起点|触发|真实失握|失握后安全完成 / 全部起点|失握后安全完成 / 真实失握|",
                  "|---|---:|---:|---:|---:|---:|"]
        for panel, both in summary["slip_recovery"].items():
            for mode, data in both.items():
                lines.append(f"|{panel} / {mode}|{ratio(data['availability'])}|{data['injected']}|{data['actual_losses']}|"
                             f"{ratio(data['recovery_all_starts'])}|{ratio(data['recovery_given_actual_loss'])}|")
        lines += ["", "### 冻结 Actor 全帧在线跟踪", "",
                  "|任务 / 物理量|分层|帧数|P95|Max|", "|---|---|---:|---:|---:|"]
        for task, data in summary["tracking"]["all_actor_trajectories"].items():
            for group, stats in data["strata"].items():
                lines.append(f"|{task} ({data['unit']})|{group}|{stats['frames']}|{fmt(stats['p95'])}|{fmt(stats['max'])}|")
        lines += ["", "全帧统计覆盖自然回合及各干预分支。逐帧文件保留原始估计、真值和可见性/接触标记，各分层可重叠。"
                  "分层中 0 帧表示本次没有得到该条件的证据。", ""]
    target = run / "STEP1B_PILOT_REPORT.md"
    temp = target.with_suffix(".tmp")
    temp.write_text("\n".join(lines))
    temp.replace(target)
