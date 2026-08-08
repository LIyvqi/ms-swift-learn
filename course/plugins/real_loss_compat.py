"""修正当前 ms-swift 开发版中原生 REAL 与动态 OPSD 探测的冲突。"""

from swift.rlhf_trainers import rollout_mixin


def 关闭本进程的动态自蒸馏(*, has_teacher_explicit: bool, is_self_distillation: bool) -> bool:
    """第 21 节没有 teacher_prompt，因此不应预先进入 OPD-RL 教师分支。"""

    return False


# 外部插件在训练器实例化前加载，替换的只是本次 Python 进程中的模块函数。
# 不改 third_party 源码，也不会影响第 20 节需要动态 OPSD 的独立训练进程。
rollout_mixin.resolve_dynamic_opd_self_distillation = 关闭本进程的动态自蒸馏
