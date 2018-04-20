import psycopg2 as pg
import config as cfg
import pandas as pd
from typing import List


class MetadataDatabaseCursor(object):

    def __enter__(self):

        """
        Establish connection to a Postgres database
        NOTE: documented problems with a lock race on a schema selection, in case of problems set time.sleep(0.1)

        :return:
        connection cursor object
        """

        self.conn = pg.connect(
            host=cfg.db_metadata['db_host'],
            port=cfg.db_metadata['db_port'],
            database=cfg.db_metadata['db_name'],
            user=cfg.db_metadata['db_username'],
            password=cfg.db_metadata['db_password']
        )

        self.cursor = self.conn.cursor()
        self.cursor.execute('SET search_path TO %s;', (cfg.db_metadata['db_schema'], ))

        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.conn.close()


class BehaviourDatabaseCursor(object):

    def __enter__(self):

        """
        Establish connection to a Postgres database
        NOTE: documented problems with a lock race on a schema selection, in case of problems set time.sleep(0.1)

        :return:
        connection cursor object
        """

        self.conn = pg.connect(
            host=cfg.db_behaviour['db_host'],
            port=cfg.db_behaviour['db_port'],
            database=cfg.db_behaviour['db_name'],
            user=cfg.db_behaviour['db_username'],
            password=cfg.db_behaviour['db_password']
        )

        self.cursor = self.conn.cursor()
        self.cursor.execute('SET search_path TO %s;', (cfg.db_behaviour['db_schema'],))

        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.conn.close()


def fetch_unique_user_cookies(limit: int or None=None):
    """
    Return unique cookies from database [behaviour.events]
    NOTE: Rewrite with yield if RAM usage issues occur
    NOTE: Possible use of a super-slow DISTINCT? Sticking to the use of Set for now.

    :param limit:
    limit number of results for fast-testing purposes

    :return:
    list of unique cookies
    """

    with BehaviourDatabaseCursor() as db_cursor:

        unique_cookies = []

        if limit is not None:
            db_cursor.execute(
                "SELECT DISTINCT cookie "
                "FROM events "
                "WHERE entitytype = 'user' "
                "LIMIT %s::BIGINT;", (limit,)
            )
        else:
            db_cursor.execute(
                "SELECT DISTINCT cookie "
                "FROM events "
                "WHERE entitytype = 'user';"
            )

        result_rows = db_cursor.fetchall()

        for row in result_rows:
            unique_cookies.append(str(row[0]))

    return unique_cookies


def fetch_employee_ids() -> set:
    """
    Select employee user_id set from database to filter out of dataset.

    :return:
    set of employee ids
    """

    employee_user_ids = set()

    with BehaviourDatabaseCursor() as db_cursor:
        db_cursor.execute(
            "SELECT user_id FROM employees;"
        )
        result_rows = db_cursor.fetchall()

        for row in result_rows:
            employee_user_ids.add(row[0])

    return employee_user_ids


def get_cookie_event_amounts(filter_minimum: int) -> []:
    """
    Fetch unique cookies and amount of events per each.

    :param filter_minimum:
    filter event amount with less than amount specified
    (users with too little events to bring any information)
    :return
    list of event amounts
    """
    user_event_amounts = []

    with BehaviourDatabaseCursor() as db_cursor:
        db_cursor.execute(
            "SELECT cookie, COUNT(event) as amount FROM events GROUP BY cookie;"
        )

        result_rows = db_cursor.fetchall()

        for row in result_rows:
            if row[1] >= filter_minimum:
                user_event_amounts.append(row[1])

    return user_event_amounts


def fetch_categories():
    """
    Get categories rows from database.
    :return:
    list of category tuples
    """

    with MetadataDatabaseCursor() as db_cursor:
        db_cursor.execute(
            "SELECT id, name, name_url, parent_id "
            "FROM category "
            "ORDER by id;"
        )

        result_rows = db_cursor.fetchall()

        return result_rows


def fetch_unique_category_ids() -> List[int]:

    unique_ids = []

    with BehaviourDatabaseCursor() as db_cursor:
        db_cursor.execute(
            "SELECT DISTINCT entityid "
            "FROM events "
            "WHERE entitytype = 'category';"
        )

        result_rows = db_cursor.fetchall()

        for row in result_rows:
            unique_ids.append(row[0])

    return unique_ids


def fetch_unique_event_types() -> List[str]:

    unique_event_types = []

    with BehaviourDatabaseCursor() as db_cursor:

        db_cursor.execute(
            "SELECT DISTINCT event FROM events;"
        )

        result_rows = db_cursor.fetchall()

    for row in result_rows:
        unique_event_types.append(row[0])

    return unique_event_types


def fetch_deals():
    """
    Fetch database deals.
    NOTE: Rewrite with yield if RAM usage issues occur
    NOTE: ID 22 seems like a test, all info is None
    :return:
    list of tuples representing database rows
    """

    with MetadataDatabaseCursor() as db_cursor:

        db_cursor.execute(
            "SELECT id, partner_id "
            "FROM deal;"
        )
        result_rows = db_cursor.fetchall()

    return result_rows


def fetch_targeted_deal_ids() -> set:

    targeted_ids = set()

    with BehaviourDatabaseCursor() as db_cursor:

        db_cursor.execute(
            "SELECT DISTINCT targetentityid "
            "FROM events "
            "WHERE targetentitytype = 'deal' AND entitytype = 'user';"
        )
        result_rows = db_cursor.fetchall()

    for row in result_rows:
        try:
            targeted_ids.add(int(row[0]))
        except TypeError:
            print(f'Could not cast {row[0]} to integer value!')
            raise TypeError

    return targeted_ids


def fetch_deal_items(db_cursor: MetadataDatabaseCursor, id_deal: int) -> tuple:
    """
    Returns important data from deal items.
    :param db_cursor:
    cursor passed instead of created to limit amount of DB connections established at runtime.
    :param id_deal:
    deal whose items will be fetched
    :return:
    database rows of targeted data
    """


    db_cursor.execute(
        "SELECT team_price "
        "FROM dealitem "
        "WHERE deal_id = %s::BIGINT;", (id_deal,)
    )

    result_rows = db_cursor.fetchall()

    return result_rows


def fetch_deal_metrics(db_cursor: MetadataDatabaseCursor, id_deal: int) -> int:

    db_cursor.execute(
        "SELECT unique_pageviews "
        "FROM deal_metrics "
        "WHERE deal_id = %s::BIGINT;", (id_deal,)
    )

    result_rows = db_cursor.fetchall()

    if len(result_rows) != 1:
        print(f'Did not fetch exactly one row for deal ID {id_deal}!')
        return 0

    return result_rows[0][0]


def fetch_deal_income(db_cursor: MetadataDatabaseCursor, id_deal: int) -> (int, int):
    """
    Fetch amount of coupon creations and cancellations from metadata database.
    :param db_cursor:
    database cursor
    :param id_deal:
    target deal ID
    :return:
    coupons_created, coupons_canceled
    """

    db_cursor.execute(
        "SELECT "
        "SUM(coupons_created) AS sum_coupons_created, "
        "SUM(coupons_canceled) AS sum_coupons_canceled "
        "FROM deal_income "
        "WHERE deal_id = %s::BIGINT;", (id_deal,)
    )

    result_rows = db_cursor.fetchall()

    coupons_created = result_rows[0][0]
    coupons_canceled = result_rows[0][1]

    return coupons_created, coupons_canceled


def fetch_deal_categories(db_cursor: MetadataDatabaseCursor, id_deal: int) -> [int]:
    """
    Fetch deal categories from metadata database.
    :param db_cursor:
    cursor passed instead of created to limit amount of DB connections established at runtime.
    :param id_deal:
    deal whose categories will be fetched
    :return:
    list of integers, category IDs
    """

    categories = set()

    db_cursor.execute(
        "SELECT category_id "
        "FROM dealcategory "
        "WHERE deal_id = %s::BIGINT;", (id_deal,)
    )
    result_rows = db_cursor.fetchall()

    [categories.add(row[0]) for row in result_rows]

    return list(categories)


def fetch_partners():
    """
    Fetch database events containing partners being $set to the website.
    NOTE: Rewrite with yield if RAM usage issues occur

    :return:
    list of tuples representing database rows
    """

    with MetadataDatabaseCursor() as db_cursor:

        db_cursor.execute(
            "SELECT id, avg_rating, rating_count "
            "FROM partner "
            "ORDER BY id;"
        )
        result_rows = db_cursor.fetchall()

    return result_rows


def fetch_events_by_cookie(cursor, cookie: str) -> tuple:
    """
    Get events bound to a specific cookie.

    NOTE: Not instancing database connection here to not open and close database every microsecond
    Make connection instance wrap usage of this function in a loop outside of a function scope

    :param cursor:
    database connection cursor for executing queries
    :param cookie:
    cookie  to search events
    :return:
    list of tuples representing database rows
    """

    cursor.execute('SELECT unique_hash, entityid, entitytype, event, eventtime, '
                   'properties, ea, targetentityid, targetentitytype '
                   'FROM events WHERE cookie = %s ORDER BY eventtime ASC;', (cookie,))

    return cursor.fetchall()


def fetch_user_info_by_cookie(cursor, cookie: str) -> (str, str) or None:
    """
    Search for a user info based on a cookie.
    If user is not registered, return None.

    NOTE: Not instancing database connection here to not open and close database every microsecond
    Make connection instance wrap usage of this function in a loop outside of a function scope

    :param cursor:
    database connection cursor for executing queries
    :param cookie:
    cookie  to search user info
    :return:
    list of tuples representing database rows
    """

    cursor.execute('SELECT user_id, created_at FROM cookies WHERE cookie_id = %s '
                   'ORDER BY created_at DESC;', (cookie,))
    results = cursor.fetchall()

    if not len(results):
        return None
    else:
        return results[0][0], results[0][1]


def get_ranges_coupons_created(db_cursor: MetadataDatabaseCursor):

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER by coupons_created_sum) "
        "FROM "
        "(	"
        "SELECT deal_id, SUM(coupons_created) as coupons_created_sum "
        "FROM metadata.deal_income "
        "GROUP BY deal_id "
        "ORDER BY coupons_created_sum DESC "
        ") AS grouped_coupons_created;"
    )

    lower_quartile = db_cursor.fetchall()[0][0]

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER by coupons_created_sum) "
        "FROM "
        "(	"
        "SELECT deal_id, SUM(coupons_created) as coupons_created_sum "
        "FROM metadata.deal_income "
        "GROUP BY deal_id "
        "ORDER BY coupons_created_sum DESC "
        ") AS grouped_coupons_created;"
    )

    upper_quartile = db_cursor.fetchall()[0][0]

    interquartile_range = upper_quartile - lower_quartile

    lower_outlier_border = lower_quartile - 1.5 * interquartile_range
    upper_outlier_border = upper_quartile + 1.5 * interquartile_range

    return max(0, lower_outlier_border), upper_outlier_border


def get_ranges_coupons_canceled(db_cursor: MetadataDatabaseCursor):

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER by coupons_canceled_sum) "
        "FROM "
        "(	"
        "SELECT deal_id, SUM(coupons_canceled) as coupons_canceled_sum "
        "FROM deal_income "
        "GROUP BY deal_id "
        "ORDER BY coupons_canceled_sum DESC "
        ") AS grouped_coupons_canceled;"
    )

    lower_quartile = db_cursor.fetchall()[0][0]

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER by coupons_canceled_sum) "
        "FROM "
        "(	"
        "SELECT deal_id, SUM(coupons_canceled) as coupons_canceled_sum "
        "FROM deal_income "
        "GROUP BY deal_id "
        "ORDER BY coupons_canceled_sum DESC "
        ") AS grouped_coupons_canceled;"
    )

    upper_quartile = db_cursor.fetchall()[0][0]

    interquartile_range = upper_quartile - lower_quartile

    lower_outlier_border = lower_quartile - 1.5 * interquartile_range
    upper_outlier_border = upper_quartile + 1.5 * interquartile_range

    return max(0, lower_outlier_border), upper_outlier_border


def get_ranges_unique_pageviews(db_cursor: MetadataDatabaseCursor):

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER by unique_pageviews) "
        "FROM "
        "(	"
        "SELECT deal_id, unique_pageviews "
        "FROM deal_metrics "
        "ORDER BY unique_pageviews DESC "
        ") AS unique_pageviews;"
    )

    lower_quartile = db_cursor.fetchall()[0][0]

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER by unique_pageviews) "
        "FROM "
        "(	"
        "SELECT deal_id, unique_pageviews "
        "FROM deal_metrics "
        "ORDER BY unique_pageviews DESC "
        ") AS unique_pageviews;"
    )

    upper_quartile = db_cursor.fetchall()[0][0]

    interquartile_range = upper_quartile - lower_quartile

    lower_outlier_border = lower_quartile - 1.5 * interquartile_range
    upper_outlier_border = upper_quartile + 1.5 * interquartile_range

    return max(0, lower_outlier_border), upper_outlier_border


def get_ranges_dealitem_prices(db_cursor: MetadataDatabaseCursor):

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER by team_price) "
        "FROM "
        "(	"
            "SELECT id, team_price "
            "FROM metadata.dealitem "
            "ORDER BY team_price DESC "
            ") AS team_prices;"
    )

    lower_quartile = db_cursor.fetchall()[0][0]

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER by team_price) "
        "FROM "
        "(	"
        "SELECT id, team_price "
        "FROM metadata.dealitem "
        "ORDER BY team_price DESC "
        ") AS team_prices;"
    )

    upper_quartile = db_cursor.fetchall()[0][0]

    interquartile_range = upper_quartile - lower_quartile

    lower_outlier_border = lower_quartile - 1.5 * interquartile_range
    upper_outlier_border = upper_quartile + 1.5 * interquartile_range

    return max(0, lower_outlier_border), upper_outlier_border


def get_ranges_partner_rating_avg(db_cursor: MetadataDatabaseCursor):

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER by avg_rating) "
        "FROM "
        "(	"
        "SELECT id, avg_rating "
        "FROM metadata.partner "
        "ORDER BY avg_rating DESC "
        ") AS avg_ratings;"
    )

    lower_quartile = db_cursor.fetchall()[0][0]

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER by avg_rating) "
        "FROM "
        "(	"
        "SELECT id, avg_rating "
        "FROM metadata.partner "
        "ORDER BY avg_rating DESC "
        ") AS avg_ratings;"
    )

    upper_quartile = db_cursor.fetchall()[0][0]

    interquartile_range = upper_quartile - lower_quartile

    lower_outlier_border = lower_quartile - 1.5 * interquartile_range
    upper_outlier_border = upper_quartile + 1.5 * interquartile_range

    return max(0, lower_outlier_border), min(upper_outlier_border, 5)


def get_ranges_partner_rating_count(db_cursor: MetadataDatabaseCursor):

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER by rating_count) "
        "FROM "
        "(	"
        "SELECT id, rating_count "
        "FROM metadata.partner "
        "ORDER BY rating_count DESC "
        ") AS rating_counts;"
    )

    lower_quartile = db_cursor.fetchall()[0][0]

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER by rating_count) "
        "FROM "
        "(	"
        "SELECT id, rating_count "
        "FROM metadata.partner "
        "WHERE rating_count <> 0"
        "ORDER BY rating_count DESC "
        ") AS rating_counts;"
    )

    upper_quartile = db_cursor.fetchall()[0][0]

    interquartile_range = upper_quartile - lower_quartile

    lower_outlier_border = lower_quartile - 1.5 * interquartile_range
    upper_outlier_border = upper_quartile + 1.5 * interquartile_range

    return max(0, lower_outlier_border), upper_outlier_border


def get_ranges_event_cookie_amount(db_cursor: BehaviourDatabaseCursor):

    lower_quartile = 0

    db_cursor.execute(
        "SELECT PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER by event_amount) "
        "FROM "
        "(	SELECT * FROM"
        "("
        "SELECT cookie, COUNT(event) as event_amount "
        "FROM events "
        "GROUP BY cookie "
        "ORDER BY event_amount DESC"
        ") AS event_amounts "
        "WHERE event_amount >= %s::BIGINT"
        ") AS limited_event_amounts;", (cfg.user_event_amount_minimum,)
    )

    upper_quartile = db_cursor.fetchall()[0][0]

    interquartile_range = upper_quartile - lower_quartile

    lower_outlier_border = lower_quartile - 1.5 * interquartile_range
    upper_outlier_border = upper_quartile + 1.5 * interquartile_range

    return max(cfg.user_event_amount_minimum, lower_outlier_border), upper_outlier_border



