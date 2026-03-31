"""PredQL pre-defined tasks."""

from .predql_stat_tasks import (
    StatsPostTagsStatTask,
    StatsUserReputationStatTask,
    StatsUserBadgeStatTask,
    StatsUserEngagementStatTask,
    StatsPostVotesStatTask,
    StatsUserPostCommentStatTask,
    StatsPostPostRelatedStatTask,

    GrantsAwardsInstitutionStatTask,
    GrantsCountInstitutionAwardsStatTask,
    GrantsOrganizationAwardsAmountStatTask,
)
from .predql_tmp_tasks import (
    StatsPostVotesTmpTask,
    StatsUserBadgeTmpTask,
    StatsUserEngagementTmpTask,
    StatsUserPostCommentTmpTask,
    StatsPostPostRelatedTmpTask,

    SeznamClientOutOfWalletTmpTask,
    SeznamClientServisTmpTask,
    SeznamClientSpendingTmpTask
)

__all__ = [
    "StatsUserReputationStatTask",
    "StatsPostTagsStatTask",
    "StatsUserBadgeStatTask",
    "StatsUserEngagementStatTask",
    "StatsPostVotesStatTask",
    "StatsUserPostCommentStatTask",
    "StatsPostPostRelatedStatTask",

    "GrantsAwardsInstitutionStatTask",
    "GrantsCountInstitutionAwardsStatTask",
    "GrantsOrganizationAwardsAmountStatTask",

    "StatsUserBadgeTmpTask",
    "StatsUserEngagementTmpTask",
    "StatsPostVotesTmpTask",
    "StatsUserPostCommentTmpTask",
    "StatsPostPostRelatedTmpTask",

    "SeznamClientOutOfWalletTmpTask",
    "SeznamClientServisTmpTask",
    "SeznamClientSpendingTmpTask"
]
