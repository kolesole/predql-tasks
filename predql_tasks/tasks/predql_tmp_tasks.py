"""Collection of pre-defined PredQL temporal tasks on ctu datasets."""

import pandas as pd
from relbench.base import TaskType
from relbench.datasets import get_dataset

from predql_tasks.base import PredQLTmpTask

######### CTU tasks ##########

######### DATASET: ctu-sfscores #########

class SFScoresBusinessesScoresTmpTask(PredQLTmpTask):
    """Predict maximum future scores for each business in the next 182 days."""

    dataset = get_dataset("ctu-sfscores", download=False)
    entity_table = "businesses"
    task_type = TaskType.MULTICLASS_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//2)
    val_timestamp = pd.Timestamp("2015-12-17")
    test_timestamp = pd.Timestamp("2016-06-16")

    predql_query = """
          PREDICT MAX(inspections.score, 0, 182, DAYS)
          FOR EACH businesses.*;
     """

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
    dst_table = "posts"

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
    dst_table = "posts"

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


class SeznamClientFirstServisTmpTask(PredQLTmpTask):
    """Predict the first sercis a client will use in the next 30 days."""

    dataset = get_dataset("ctu-seznam", download=False)
    entity_table = "client"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    predql_query = """
          PREDICT FIRST(probehnuto.sluzba, 0, 30, DAYS)
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

######### RelBench tasks (defined with PredQL) #########

######### DATASET: rel-f1 #########

class RelF1DriverDNFTmpTask(PredQLTmpTask):
    """For each driver predict the if they will DNF (did not finish) a race in the next 1 month."""

    dataset = get_dataset("rel-f1", download=False)
    entity_table = "drivers"
    task_type = TaskType.BINARY_CLASSIFICATION
    num_eval_timestamps = 40

    timedelta = pd.Timedelta(days=30)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT MAX(results.statusId, 0, 30, DAYS) != 1
          FOR EACH drivers.*
          ASSUMING COUNT(results.*, -365, 0, DAYS) != 0
          WHERE MAX(results.statusID, 0, 30, DAYS) IS NOT NULL;
     """


class RelF1DriverTop3TmpTask(PredQLTmpTask):
    """For each driver predict if they will qualify in the top-3 for a race in the next 1 month."""

    dataset = get_dataset("rel-f1", download=False)
    entity_table = "drivers"
    task_type = TaskType.BINARY_CLASSIFICATION
    num_eval_timestamps = 40

    timedelta = pd.Timedelta(days=30)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT MIN(qualifying.position, 0, 30, DAYS) <= 3
          FOR EACH drivers.*
          WHERE MIN(qualifying.position, 0, 30, DAYS) IS NOT NULL;
     """


class RelF1DriverPositionTmpTask(PredQLTmpTask):
    """Predict the average finishing position of each driver all races in the next 2 months."""

    dataset = get_dataset("rel-f1", download=False)
    entity_table = "drivers"
    task_type = TaskType.REGRESSION
    num_eval_timestamps = 40

    timedelta = pd.Timedelta(days=60)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT AVG(results.positionOrder, 0, 60, DAYS)
          FOR EACH drivers.*;
     """

######### DATASET: rel-stack #########

class RelStackUserEngagementTmpTask(PredQLTmpTask):
    """For each user predict if a user will make any votes, posts, or comments in the next 3 months."""

    dataset = get_dataset("rel-stack", download=False)
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT COUNT(votes.*, 0, 91, DAYS) != 0
               OR COUNT(posts.*, 0, 91, DAYS) != 0
               OR COUNT(comments.*, 0, 91, DAYS) != 0
          FOR EACH users.*
          ASSUMING COUNT(votes.*, -inf, 0, DAYS) != 0
               OR COUNT(posts.*, -inf, 0, DAYS) != 0
               OR COUNT(comments.*, -inf, 0, DAYS) != 0;
     """


class RelStackUserBadgeTmpTask(PredQLTmpTask):
    """For each user predict if a user will receive a new badge in the next 3 months."""

    dataset = get_dataset("rel-stack", download=False)
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT COUNT(badges.*, 0, 91, DAYS) != 0
          FOR EACH users.*;
     """


class RelStackPostVotesTmpTask(PredQLTmpTask):
    """For each user post predict how many votes it will receive in the next 3 months."""

    dataset = get_dataset("rel-stack", download=False)
    entity_table = "posts"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT COUNT_DISTINCT(votes.* WHERE votes.votetypeid == 2, 0, 91, DAYS)
          FOR EACH posts.* WHERE posts.PostTypeId == 1
                             AND posts.OwnerUserId IS NOT NULL
                             AND posts.OwnerUserId != -1;
     """


class RelStackUserPostCommentTmpTask(PredQLTmpTask):
    """Predict a list of existing posts that a user will comment in the next two months."""

    dataset = get_dataset("rel-stack", download=False)
    entity_table = "users"
    task_type = TaskType.LINK_PREDICTION
    dst_table = "posts"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT LIST_DISTINCT(comments.PostId
               WHERE posts.owneruserid IS NOT NULL
                    AND posts.owneruserid != -1, 0, 91, DAYS)
          FOR EACH users.*;
     """


class RelStackPostPostRelatedTmpTask(PredQLTmpTask):
    """Predict a list of existing posts that users will link a given post to in the next two months."""

    dataset = get_dataset("rel-stack", download=False)
    entity_table = "posts"
    task_type = TaskType.LINK_PREDICTION
    dst_table = "posts"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    predql_query = """
          PREDICT LIST_DISTINCT(postLinks.RelatedPostId, 0, 91, DAYS)
          FOR EACH posts.*;
     """
