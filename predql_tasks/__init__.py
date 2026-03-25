from predql_tasks.predql_base_task import PredQLBaseTask
from predql_tasks.predql_task_stat import PredQLTaskStat
from predql_tasks.predql_task_tmp import PredQLTaskTmp
from predql_tasks.predql_tasks import (
    StatsPostTagsStatTask,
    StatsUserReputationStatTask,
    StatsUserBadgeStatTask,
    StatsUserEngagementStatTask,
    StatsPostVotesStatTask,
    StatsUserPostCommentStatTask,
    StatsPostPostRelatedStatTask,

    StatsPostVotesTmpTask,
    StatsUserBadgeTmpTask,
    StatsUserEngagementTmpTask,
    StatsUserPostCommentTmpTask,
    StatsPostPostRelatedTmpTask,

    GrantsAwardsInstitutionStatTask,
    GrantsCountInstitutionAwardsStatTask,
    GrantsOrganizationAwardsAmountStatTask,
    # TestTask

    SeznamClientOutOfWalletTmpTask,
    SeznamClientServisTmpTask,
    SeznamClientSpendingTmpTask
)

__all__ = [
    "PredQLBaseTask",
    "PredQLTaskStat",
    "PredQLTaskTmp",

    "StatsUserReputationStatTask",
    "StatsPostTagsStatTask",
    "StatsUserBadgeStatTask",
    "StatsUserEngagementStatTask",
    "StatsPostVotesStatTask",
    "StatsUserPostCommentStatTask",
    "StatsPostPostRelatedStatTask",

    "StatsUserBadgeTmpTask",
    "StatsUserEngagementTmpTask",
    "StatsPostVotesTmpTask",
    "StatsUserPostCommentTmpTask",
    "StatsPostPostRelatedTmpTask",

    "GrantsAwardsInstitutionStatTask",
    "GrantsCountInstitutionAwardsStatTask",
    "GrantsOrganizationAwardsAmountStatTask",
    # "TestTask"

    "SeznamClientOutOfWalletTmpTask",
    "SeznamClientServisTmpTask",
    "SeznamClientSpendingTmpTask"
]
