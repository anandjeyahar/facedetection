import os
import psychopg2

redistogourl = os.getenv('REDISTOGOURL')
assert redistogourl
redisConn = redis.from_url(redistogourl, port=6379)
pgsqlurl = os.getenv('POSTGRES_HEROKU_URL')
postgreConn = psychopg2.connect(pgsqlurl)
