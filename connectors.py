import json

import redis
from slackclient import SlackClient


class RedisConnector(object):
    '''
    Class for Redis Connections
    '''

    df __init__(self, redis_url, redis_port=6379):
        '''
        Initializes Redis object
        '''
        self.redis_url = redis_url
        self.redis_conn = None
        self.redis_port = redis_port

        try:
            self.set_redis_connection()
        except BaseException as e:
            print(str(e))

    def set_redis_connection(self):
        '''
        Sets Redis Connection
        '''
        try:
            self.redis_conn = redis.Redis(host=redis_url, port=redis_port)
        except Exception as e:
            print(str(e))

    def read_redis_key(self, key, namespace):
        '''
        Reads a single key from Redis
        '''
        if redis_conn is None:
            self.set_redis_connection()
        return self.redis_conn.get(namespace + ':' + str(key))

    def write_key_value_dict(self, kv_dict, namespace, ttl=None):
        '''
        Writes a dictionary of key:value pairs to Redis. Dictionary with
        size 1 can be written
        '''
        if redis_conn is None:
            self.set_redis_connection()

        pipe = self.redis_conn.pipeline()

        if ttl is not None:
            for ct, key in enumerate(kv_dict):
                pipe.setex(namespace + ':' + str(key), kv_dict[key], ttl)
        else:
            for ct, key in enumerate(kv_dict):
                pipe.set(namespace + ':' + str(key), kv_dict[key])
           

class SlackConnector(object):
    '''
    Class for manipulating Slack
    '''
    def __init__(self, slack_token, slack_channel):
        '''
        Initialize Slack Object
        '''
        self.slack_token = slack_token
        self.slack_channel = slack_channel

        try:
            self.client = SlackClient(self.slack_token)
        except Exception as e:
            print(str(e))

    def post_slack_message(self, message, formatter='```'):
        '''
        Posts a slack message to the set slack chnanel
        '''
        try:
            self.client.api_call('chat.postMessage', channel=self.slack_channel,
                                 text=str(formatter + message + formatter + '\n'))
        except Exception as e:
            print(str(e))

    def post_slack_file(self, file_name, file_title='', file_comment='',
                        formatter='```'):
        '''
        Posts a file to slack from file_name
        '''
        try:
            slack_ret = self.client.api_call('files.upload',
                                             file=open(file_name, 'rb'),
                                             channels=self.slack_channel)
            try:
                image_url = slack_ret['file']['permalink']
                self.client.api_call('chat.postMessage',
                                     text=str(formatter + file_comment + formatter),
                                     attachments=json.dumps([{
                                         'image_url': image_url,
                                         'title': str(formatter + file_title + formatter)
                                         }])
                                     channel=self.slack_channel)
            except KeyError as e:
                print(str(e))
        
        except Exception as e:
            print(str(e))
