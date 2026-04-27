"""PredQL pre-defined tasks."""

from .predql_stat_tasks import (
    GrantsAwardsInstitutionStatTask,
    GrantsCountInstitutionAwardsStatTask,
    GrantsOrganizationAwardsAmountStatTask,
    StatsPostPostRelatedStatTask,
    StatsPostTagsStatTask,
    StatsPostVotesStatTask,
    StatsUserBadgeStatTask,
    StatsUserEngagementStatTask,
    StatsUserPostCommentStatTask,
    StatsUserReputationStatTask,
)
from .predql_tmp_tasks import (
    ######### RelBench tasks (defined with PredQL)
    RelF1DriverDNFTmpTask,
    RelF1DriverPositionTmpTask,
    RelF1DriverTop3TmpTask,
    RelStackPostPostRelatedTmpTask,
    RelStackPostVotesTmpTask,
    RelStackUserBadgeTmpTask,
    RelStackUserEngagementTmpTask,
    RelStackUserPostCommentTmpTask,
    SeznamClientFirstServisTmpTask,
    SeznamClientOutOfWalletTmpTask,
    SeznamClientServisTmpTask,
    SeznamClientSpendingTmpTask,
    SFScoresBusinessesScoresTmpTask,
    StatsPostPostRelatedTmpTask,
    StatsPostVotesTmpTask,
    StatsUserBadgeTmpTask,
    StatsUserEngagementTmpTask,
    StatsUserPostCommentTmpTask,
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

    "SFScoresBusinessesScoresTmpTask",

    "StatsUserBadgeTmpTask",
    "StatsUserEngagementTmpTask",
    "StatsPostVotesTmpTask",
    "StatsUserPostCommentTmpTask",
    "StatsPostPostRelatedTmpTask",

    "SeznamClientOutOfWalletTmpTask",
    "SeznamClientServisTmpTask",
    "SeznamClientFirstServisTmpTask",
    "SeznamClientSpendingTmpTask",

    "RelF1DriverDNFTmpTask",
    "RelF1DriverTop3TmpTask",
    "RelF1DriverPositionTmpTask",

    "RelStackUserEngagementTmpTask",
    "RelStackUserBadgeTmpTask",
    "RelStackPostVotesTmpTask",
    "RelStackUserPostCommentTmpTask",
    "RelStackPostPostRelatedTmpTask"
]
