import os
import psycopg2
import redis
import urlparse

urlparse.uses_netloc.append("postgres")
url = urlparse.urlparse(os.environ["POSTGRES_HEROKU_URL"])

conn = psycopg2.connect(
    database=url.path[1:],
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port
)

redistogourl = os.getenv('REDISTOGOURL')
assert redistogourl
redisConn = redis.from_url(redistogourl, port=6379)
pgsqlurl = os.getenv('POSTGRES_HEROKU_URL')
postgreConn = psycopg2.connect(pgsqlurl)
