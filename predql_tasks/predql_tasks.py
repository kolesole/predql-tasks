import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import TaskType

from predql_tasks import PredQLTaskStat, PredQLTaskTmp

######### DATASET: сtu-stats #########

######### STATIC TASKS       #########

class StatsUserReputationStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.REGRESSION

    predql_query = """
          PREDICT users.reputation
          FOR EACH users.*
          WHERE COUNT(votes.*) != 0
             OR COUNT(comments.*) != 0
             OR COUNT(posts.*) != 0;
     """


class StatsPostTagsStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.MULTILABEL_CLASSIFICATION

    predql_query = """
          PREDICT LIST_DISTINCT(tags.*)
          FOR EACH posts.*;
     """


class StatsUserBadgeStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.REGRESSION

    predql_query = """
          PREDICT COUNT(badges.*)
          FOR EACH users.*;
     """


class StatsUserEngagementStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    predql_query = """
          PREDICT COUNT(votes.*) != 0
               OR COUNT(posts.*) != 0
               OR COUNT(comments.*) != 0
          FOR EACH users.*;
     """


class StatsPostVotesStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.REGRESSION
     
    predql_query = """
          PREDICT COUNT_DISTINCT(votes.* 
               WHERE votes.votetypeid == 2)
          FOR EACH posts.* WHERE posts.PostTypeId == 1
                             AND posts.OwnerUserId IS NOT NULL
                             AND posts.OwnerUserId != -1;
      """


class StatsUserPostCommentStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.LINK_PREDICTION

    predql_query = """
          PREDICT LIST_DISTINCT(comments.FK_posts_PostId
               WHERE posts.owneruserid IS NOT NULL
                 AND posts.owneruserid != -1)
          FOR EACH users.*;
     """


class StatsPostPostRelatedStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.LINK_PREDICTION

    predql_query = """
          PREDICT LIST_DISTINCT(postLinks.FK_posts_PostId)
          FOR EACH posts.*;
     """

######### TEMPORAL TASKS     #########

class StatsUserBadgeTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    predql_query = """
          PREDICT COUNT(badges.*, 0, 91, DAYS) != 0
          FOR EACH users.*;
     """


class StatsUserEngagementTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    predql_query = """
          PREDICT COUNT(votes.*, 0, 91, DAYS) != 0
               OR COUNT(posts.*, 0, 91, DAYS) != 0
               OR COUNT(comments.*, 0, 91, DAYS) != 0
          FOR EACH users.*
          ASSUMING COUNT(votes.*, -inf, 0, SECONDS) != 0
               OR COUNT(posts.*, -inf, 0, SECONDS) != 0
               OR COUNT(comments.*, -inf, 0, SECONDS) != 0;
     """


class StatsPostVotesTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    predql_query = """
          PREDICT COUNT_DISTINCT(votes.* 
               WHERE votes.votetypeid == 2, 0, 91, DAYS)
          FOR EACH posts.* WHERE posts.PostTypeId == 1
                             AND posts.OwnerUserId IS NOT NULL
                             AND posts.OwnerUserId != -1;
     """


class StatsUserPostCommentTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.LINK_PREDICTION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    predql_query = """
          PREDICT LIST_DISTINCT(comments.FK_posts_PostId 
               WHERE posts.owneruserid IS NOT NULL
                 AND posts.owneruserid != -1, 0, 91, DAYS)
          FOR EACH users.*;
     """


class StatsPostPostRelatedTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.LINK_PREDICTION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    predql_query = """
          PREDICT LIST_DISTINCT(postLinks.FK_posts_RelatedPostId, 0, 91, DAYS)
          FOR EACH posts.*;
     """
    

######### DATASET: сtu-grants #########

######### STATIC TASKS        #########

class GrantsAwardsInstitutionStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-grants", download=False)
    entity_table = "awards"
    task_type = TaskType.MULTICLASS_CLASSIFICATION

    predql_query = """
          PREDICT institution_awards.name
          FOR EACH awards.*;
     """


class GrantsCountInstitutionAwardsStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-grants", download=False)
    entity_table = "institution"
    task_type = TaskType.REGRESSION

    predql_query = """
          PREDICT COUNT_DISTINCT(institution_awards.*)
          FOR EACH institution.*;
     """
    

class GrantsOrganizationAwardsAmountStatTask(PredQLTaskStat):
    dataset = get_dataset("ctu-grants", download=False)
    entity_table = "organization"
    task_type = TaskType.REGRESSION

    predql_query = """
          PREDICT SUM(awards.award_amount)
          FOR EACH organization.*;
     """
    

######### TEMPORAL TASKS      #########

# class TestTask(PredQLTaskTmp):
#     dataset = get_dataset("ctu-grants", download=False)

#     timedelta = pd.Timedelta(days=365//4)
#     val_timestamp = pd.Timestamp("2010-10-01")
#     test_timestamp = pd.Timestamp("2014-01-01")

#     predql_query = """
#           PREDICT COUNT(institution_awards.*, 0, 91, DAYS)
#           FOR EACH institution_awards.FK_awards_award_id;
#      """

######### DATASET: сtu-seznam #########

######### STATIC TASKS        #########

######### TEMPORAL TASKS      #########

class SeznamClientOutOfWalletTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-seznam", download=False)
    entity_table = "client"
    task_type = TaskType.BINARY_CLASSIFICATION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    predql_query = """
          PREDICT COUNT(probehnuto_mimo_penezenku.*, 0, 30, DAYS) != 0
          FOR EACH client.*
          ASSUMING COUNT(probehnuto.*, -inf, 0, SECONDS) != 0
                OR COUNT(dobito.*, -inf, 0, SECONDS) != 0
                OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, SECONDS) != 0
          WHERE COUNT(probehnuto.*, 0, 30, DAYS) == 0;
     """


class SeznamClientServisTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-seznam", download=False)
    entity_table = "client"
    task_type = TaskType.MULTILABEL_CLASSIFICATION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    predql_query = """
          PREDICT LIST_DISTINCT(probehnuto.sluzba, 0, 30, DAYS)
          FOR EACH client.*
          ASSUMING COUNT(probehnuto.*, -inf, 0, SECONDS) != 0
                OR COUNT(dobito.*, -inf, 0, SECONDS) != 0
                OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, SECONDS) != 0;
     """


class SeznamClientSpendingTmpTask(PredQLTaskTmp):
    dataset = get_dataset("ctu-seznam", download=False)
    entity_table = "client"
    task_type = TaskType.REGRESSION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    predql_query = """
          PREDICT SUM(probehnuto.kc_proklikano, 0, 30, DAYS)
          FOR EACH client.*
          ASSUMING COUNT(probehnuto.*, -inf, 0, SECONDS) != 0
                OR COUNT(dobito.*, -inf, 0, SECONDS) != 0
                OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, SECONDS) != 0;
     """
