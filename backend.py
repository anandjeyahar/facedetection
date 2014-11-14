import os
redistogourl = os.environ('REDISTOGOURL')
assert redistogourl
redisConn = redis.Redis(redistogourl, port=6379)
