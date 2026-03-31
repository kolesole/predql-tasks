"""Collection of pre-defined PredQL temporal tasks on ctu datasets."""

import pandas as pd

from relbench.datasets import get_dataset
from relbench.base import TaskType

from predql_tasks.base import PredQLTmpTask


######### DATASET: сtu-stats #########

class StatsUserBadgeTmpTask(PredQLTmpTask):
    """Predict whether a user earns any badge in the next 91 days."""
    
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


class StatsUserEngagementTmpTask(PredQLTmpTask):
    """Predict whether a user is active in the next 91 days."""
    
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
          ASSUMING COUNT(votes.*, -inf, 0, DAYS) != 0
               OR COUNT(posts.*, -inf, 0, DAYS) != 0
               OR COUNT(comments.*, -inf, 0, DAYS) != 0;
     """


class StatsPostVotesTmpTask(PredQLTmpTask):
    """Predict future upvote count for each valid question post."""
    
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


class StatsUserPostCommentTmpTask(PredQLTmpTask):
    """Predict posts each user will comment on in the next 91 days."""

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


class StatsPostPostRelatedTmpTask(PredQLTmpTask):
    """Predict related posts linked from each post in the next 91 days."""

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

######### DATASET: сtu-seznam #########

class SeznamClientOutOfWalletTmpTask(PredQLTmpTask):
    """Predict whether a client will spend outside wallet in the next 30 days."""

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
          ASSUMING COUNT(probehnuto.*, -inf, 0, DAYS) != 0
                OR COUNT(dobito.*, -inf, 0, DAYS) != 0
                OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, DAYS) != 0
          WHERE COUNT(probehnuto.*, 0, 30, DAYS) == 0;
     """


class SeznamClientServisTmpTask(PredQLTmpTask):
    """Predict services a client will use in the next 30 days."""

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
          ASSUMING COUNT(probehnuto.*, -inf, 0, DAYS) != 0
                OR COUNT(dobito.*, -inf, 0, DAYS) != 0
                OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, DAYS) != 0;
     """


class SeznamClientSpendingTmpTask(PredQLTmpTask):
    """Predict client spending amount in the next 30 days."""

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
          ASSUMING COUNT(probehnuto.*, -inf, 0, DAYS) != 0
                OR COUNT(dobito.*, -inf, 0, DAYS) != 0
                OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, DAYS) != 0;
     """
