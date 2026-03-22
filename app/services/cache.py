import redis
import json
from app.config import settings

redisClient = redis.Redis(host= settings.REDIS_HOST, port = settings.REDIS_PORT)

def get_cache(key):
    data = redisClient.get(key)
    return json.loads(data) if data else None

def set_cache(key,value):
    redisClient.set(key, json.dumps(value),ex = 3600)