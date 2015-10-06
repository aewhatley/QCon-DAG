#!/usr/bin/env python
"""
Map-Reduce classes

Derived from mincemeat.py by Michael Fairley

The principal classes exposed are:

Server 
-------
Application should create an instance of this class and call that instance
to start the server. Multiple clients may also be started on the local 
host.

The server responds to SIGTERM (15) by draining the clients and then ending.

Server will write it's process id to the file designated by the MAPREDUCE_PID_FILE 
environment variable.

Client
-------
Application may create an instance of this class to run client(s) locally
or this module may be run as a main to start client(s) on a separate system
or as a separate process.

Clients
-------
Start multiple clients on the same machine using python multiprocessing
"""

__author__ = """Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__version__ = "0.1.5"

__all__ = ["Clients", "Client", "Server", "MapReduceError", "run_client"]

# **************************************************
# Copyright (c) 2013 Ralph W. Crosby, Texas A&M University, College Station, Texas
# Copyright (c) 2010 Michael Fairley
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# **************************************************

import argparse
import asynchat
import asyncore
import binascii
import hashlib
import hmac
import logging
from   multiprocessing import Process, cpu_count
import os
import pickle
from   pprint import pprint as pp
import random
import signal
import socket
import traceback
import sys
import time

from Utilities.Logger   import Logger, ElapsedTime, FMT_LONG
from Utilities.PickleFn import Fn2Tuple, Tuple2Fn

# **************************************************

DEFAULT_PORT     = 11235
DEFAULT_HOST     = 'localhost'
DEFAULT_PASSWORD = 'mapreduce'
DEFAULT_TIMEOUT  = 30
DEFAULT_LOG_LEVEL= logging.INFO

STOP_SIGNAL      = signal.SIGTERM

# **************************************************

class Protocol(asynchat.async_chat):

    def __init__(self, 
                 conn=None, 
                 map=None, 
                 commands={}):

        if conn:
            asynchat.async_chat.__init__(self, conn, map=map)
        else:
            asynchat.async_chat.__init__(self, map=map)

        self.set_terminator(b"\n")
        self.buffer = []
        self.auth = None
        self.mid_command = False
        self.commands = commands

    def collect_incoming_data(self, data):
        self.buffer.append(data)

    def found_terminator(self):
        if not self.auth == b"Done":
            self.logger.debug("-> %s" % b''.join(self.buffer))
            command, data = (b''.join(self.buffer).split(b":", 1))
            self._process_unauthed_command(command, data)
        elif not self.mid_command:
            self.logger.debug("-> %s" % b''.join(self.buffer))
            command, length = (b''.join(self.buffer)).split(b":", 1)
            if command == b"challenge":
                self.process_command(command, length)
            elif length:
                self.set_terminator(int(length))
                self.mid_command = command
            else:
                self.process_command(command)
        else: # Read the data segment from the previous command
            if not self.auth == b"Done":
                raise MapReduceError("Received pickled data from unauthed source")
            data = pickle.loads(b''.join(self.buffer))
            self.set_terminator(b"\n")
            command = self.mid_command
            self.mid_command = None
            self.process_command(command, data)
        self.buffer = []

    def handle_close(self, *args):
        self.logger.debug("Closing Port")
        try:
            self.close()
        except:
            exc_info = sys.exc_info()
            log.exception("{}".format(exc_info[1]))
        self.logger.debug("Port Closed")

    def process_command(self, command, data=None):

        commands = { b'challenge':  self._respond_to_challenge,
                     b'disconnect': self.handle_close }

        if command in self.commands:
            self.commands[command](command, data)
        elif command in commands:
            commands[command](command, data)
        else:
            self.handle_close()
            raise MapReduceError("Unknown command received: %s" % (command))

    def send_challenge(self, clientId):
        self.auth = binascii.hexlify(os.urandom(20))
        self.send_command(b":".join([b"challenge", str.encode(str(clientId)), self.auth]))

    def send_command(self, command, data=None):
        self.logger.debug( "<- %s" % command)
        if not b":" in command:
            command += b":"
        if data:
            pdata = pickle.dumps(data)
            command += bytes(str(len(pdata)), 'utf-8')
            self.push(command + b"\n" + pdata)
        else:
            self.push(command + b"\n")

    def _process_unauthed_command(self, command, data=None):
        commands = {
            b'challenge':  self._respond_to_challenge,
            b'auth':       self._verify_auth,
            b'disconnect': lambda x, y: self.handle_close(),
            }

        if command in commands:
            commands[command](command, data)
        else:
            self.handle_close()
            raise MapReduceError("Unknown command received: %s" % (command))
        
    def _respond_to_challenge(self, command, data):
        id, data = data.split(b":", 1)
        self.clientId = int(id)
        mac = hmac.new(self.password, data, hashlib.sha1)
        self.send_command(b":".join([b"auth", str.encode(str(self.clientId)), binascii.hexlify(mac.digest())]))
        self.post_auth_init()

    def _verify_auth(self, command, data):
        id, data = data.split(b":", 1)
        mac = hmac.new(self.password, self.auth, hashlib.sha1)
        if data == binascii.hexlify(mac.digest()):
            self.auth = b"Done"
            self.logger.debug("Client {} Authenticated other end".format(self.clientId))
        else:
            self.handle_close()
            raise MapReduceError("Authentication Failure")

# **************************************************

class Client(Process, Protocol):

    def __init__(self,
                 server   = None,
                 password = None,
                 port     = None,
                 timeout  = None,
                 loglevel = None):

        self.server   = server   if server   else DEFAULT_HOST
        self.port     = port     if port     else DEFAULT_PORT
        self.password = password if password else DEFAULT_PASSWORD
        self.timeout  = timeout  if timeout  else DEFAULT_TIMEOUT
        self.loglevel = loglevel if loglevel else DEFAULT_LOGLEVEL

        commands = { b'mapfn':     self._set_mapfn,
                     b'collectfn': self._set_collectfn,
                     b'reducefn':  self._set_reducefn,
                     b'map':       self._call_mapfn,
                     b'reduce':    self._call_reducefn }

        Process.__init__(self)
        Protocol.__init__(self,commands=commands)

        self.mapfn = None
        self.reducefn = None
        self.collectfn = None
        
    def run(self):

        signal.signal(STOP_SIGNAL, signal.SIG_IGN)   # Ignore at the clients

        self.logger = logging.getLogger(self.name)
        self.logger.addFilter(ElapsedTime())
        self.logger.setLevel(self.loglevel)

        if type(self.password) == str:
            self.password = self.password.encode("utf-8")

        self.logger.debug("Waiting to connect")

        while self.timeout:
            
            try:
                self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(None)
                self.connect((self.server, self.port))
                break

            except ConnectionRefusedError:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    self.logger.warning("SIGINT Received")
                    self.close()
                    return

                self.timeout -= 1.0
                self.close()

            except ConnectionResetError as exc:
                self.close()
                self.logger.info(exc)
                return

            except:
                self.logger.error(sys.exc_info()[1])
                raise

        else:
            self.logger.warning("Connection timeout exhausted")
            return

        self.logger.debug("Server connected")

        asyncore.loop()

    def post_auth_init(self):
        if not self.auth:
            self.send_challenge(self.clientId)

    def _set_mapfn(self, command, data):
        self.mapfn = Tuple2Fn(data)

    def _set_collectfn(self, command, data):
        self.collectfn = Tuple2Fn(data)

    def _set_reducefn(self, command, data):
        self.reducefn = Tuple2Fn(data)

    def _call_mapfn(self, command, data):

        self.logger.info("Task {} Mapping {!s}".format(self.clientId, data[0]))

        results = {}

        try:
            for k, v in self.mapfn(data[0], data[1], logger=self.logger, clientId=self.clientId):
                if k not in results:
                    results[k] = []
                results[k].append(v)
        except:
            self.logger.error(sys.exc_info())
            traceback.print_tb(sys.exc_info()[2])
            self.send_command(b'mapfail', (data[0], {}))
            return

        if self.collectfn:
            for k in results:
                results[k] = [self.collectfn(k, results[k])]

        self.send_command(b'mapdone', (data[0], results))
        self.logger.info("Task {} Mapping {!s} Finished".format(self.clientId, data[0]))

    def _call_reducefn(self, command, data):

        self.logger.info("Task {} Reducing {!s}".format(self.clientId,data[0]))

        try:
            results = self.reducefn(data[0], data[1], logger=self.logger, clientID=self.clientId)
        except:
            self.logger.error(sys.exc_info())
            traceback.print_tb(sys.exc_info()[2])
            self.send_command(b'reducefail', (data[0], None))
        else:
            self.send_command(b'reducedone', (data[0], results))
            self.logger.info("Task {} Reducing {!s} Finished".format(self.clientId,data[0]))

# **************************************************
        
class Clients(object):
    """Run a set of clients as separate processes:

    nclients is specified:
    None : Run the number of clients sized to the number of cores (default)
    0    : Run none
    >0   : Run the pool of processes sized as specified
    """

    def __init__(self, 
                 nclients = None,
                 server   = None,
                 password = None,
                 port     = None,
                 timeout  = None,
                 logger   = None,
                 loglevel = None):

        self.server   = server   if server   else DEFAULT_HOST
        self.port     = port     if port     else DEFAULT_PORT
        self.password = password if password else DEFAULT_PASSWORD
        self.timeout  = timeout  if timeout  else DEFAULT_TIMEOUT
        self.nclients = nclients
        self.logger   = logger   if logger   else logging.getLogger()
        self.loglevel = loglevel if loglevel else DEFAULT_LOG_LEVEL

    def __call__(self, **kwargs):
        """Start the clients in parallel"""

        [setattr(self, k, kwargs.pop(k)) for k in list(kwargs) if k in self.__dict__]
        if len(kwargs):
            raise TypeError("{} got one or more unexpected arguments {}".format("Clients.__call__", 
                                                                                list(kwargs.keys())))

        if self.nclients == 0:
            return

        if not self.nclients:
            self.nclients = cpu_count()

        if not self.logger:
            self.logger = Logger(fmt=FMT_LONG,
                                 name='Clients',
                                 level=self.loglevel)


        self.logger.info("Starting {} Clients".format(self.nclients))

        opts = {k: self.__dict__[k] for k in ('server', 'password', 'port', 'timeout', 'loglevel')}

        self.clients = []

        try:
            for i in range(self.nclients):

                client = Client(**opts)
                client.start()
                self.clients.append(client)

        except:
            exc_info = sys.exc_info()
            self.logger.exception("{}".format(exc_info[1]))
            return 1

        return 0

    def terminate(self):
        """Forceably terminate the clients"""
        [c.terminate for c in self.clients if c.is_alive()]

    def join(self):
        """Wait for client completion"""
        self.logger.debug("Waiting for clients to complete")

        try:
            for c in self.clients:
                c.join()
                self.logger.debug("Client ended with rc {}", c.exitcode)
        except:
            exc_info = sys.exc_info()
            self.logger.exception("{}".format(exc_info[1]))

        self.logger.info("All Clients Completed")

# **************************************************

class MapReduceError(Exception):
    
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
        
# **************************************************

class Server(asyncore.dispatcher, object):
    """Run the map/reduce server

    It is recommended that all the appropriate parameters be passed to
    the constructor.
    If nclients is not specified a set of client prcesses will be started
    equal to the number of CPU's available on the local machine.
    """

    def __init__(self, datasource, mapfn, reducefn,
                 collectfn=None,
                 port=None,
                 password=None,
                 logger= None,
                 loglevel=None,
                 nclients=None):

        self.port     = port     if port     else DEFAULT_PORT
        self.password = password if password else DEFAULT_PASSWORD
        self.logger   = logger
        self.nclients = nclients
        self.loglevel = loglevel if loglevel else DEFAULT_LOG_LEVEL

        self.socket_map = {}
        asyncore.dispatcher.__init__(self, map=self.socket_map)

        self.mapfn      = mapfn
        self.reducefn   = reducefn
        self.collectfn  = collectfn
        self.datasource = datasource
        self._fail      = False

    def __call__(self, **kwargs):
        """Run the server

        While parameters may be passed to this method, the intention is that
        the constructor will be used to create a closure setting the arguments.
        This provides a means for the user to adjust the parameters prior to 
        calling the created object to run the server.
        """

        # Process arguments

        [setattr(self, k, kwargs.pop(k)) for k in list(kwargs) if k in self.__dict__]
        if len(kwargs):
            raise TypeError("{} got one or more unexpected arguments {}".format("Clients.__call__", 
                                                                                list(kwargs.keys())))

        if type(self.password) == str:
            self.password = self.password.encode("utf-8")

        # Setup logging

        if not self.logger:
            self.logger = Logger(fmt=FMT_LONG,
                                 name='Server',
                                 level=self.loglevel)

        # Save server pid

        if 'MAPREDUCE_PID_FILE' in os.environ:
            with open(os.environ['MAPREDUCE_PID_FILE'],'w') as pf:
                print(os.getpid(), file=pf)

        # Setup signal handling

        def SIGHANDLER(sig, frame):
            "Handler for signal, just sets flag to close processing"
            self.logger.warning("Signal {} detected, waiting for clients to finish".format(sig))
            self.fail = True

        signal.signal(STOP_SIGNAL, SIGHANDLER)

        # Start Clients

        if self.nclients != 0:
            c= Clients(nclients=self.nclients,
                       password=self.password,
                       port=self.port,
                       logger=self.logger,
                       loglevel=self.loglevel)
            if c():
                return 1

        # Start Server

        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bind(("", self.port))
        self.listen(5)

        try:
            asyncore.loop(map=self.socket_map)
        except:
            self.close()
            raise
        
        # Close any remaining clients
        if self.nclients != 0:
            c.terminate()    

        if self.fail:
            self.logger.warning("Server ending abnormally")
            raise MapReduceError("Server ending abnormally")
            
        return self.taskmanager.results

    _clientId = 0

    def handle_accepted(self, conn, addr):
        sc = ServerChannel(conn, addr, self.socket_map, Server._clientId, self)
        Server._clientId += 1

    def handle_close(self):
        self.close()

    def set_datasource(self, ds):
        self._datasource = ds
        self.taskmanager = TaskManager(self._datasource, self)
    
    def get_datasource(self):
        return self._datasource

    datasource = property(get_datasource, set_datasource)

    def get_fail(self):
        return self._fail

    def set_fail(self, v):
        self._fail = v
        if v:
            self.taskmanager.state = TaskManager.FINISHED

    fail = property(get_fail, set_fail)
    
# **************************************************

class ServerChannel(Protocol):
    """Handle connection to a client"""

    def __init__(self, conn, addr, map, clientId, server):

        commands = {
            b'mapdone'   : self._map_done,
            b'reducedone': self._reduce_done,
            b'mapfail'   : self._map_fail,
            b'reducefail': self._reduce_fail,
            }

        self.server   = server
        self.password = server.password
        self.logger   = server.logger
        self.last     = False
        self.addr     = addr
        self.clientId = clientId

        super().__init__(conn, map, commands)

        self.send_challenge(self.clientId)

    def handle_close(self):
        self.logger.info("Client {0} disconnected: {1[0]}:{1[1]}".format(self.clientId,self.addr))
 
        super().handle_close()
        if self.last:
            self.server.close()

    def post_auth_init(self):

        self.logger.info("Client {0} connected: {1[0]}:{1[1]}".format(self.clientId,self.addr))

        if self.server.mapfn:
            self.send_command(b'mapfn', Fn2Tuple(self.server.mapfn))

        if self.server.reducefn:
            self.send_command(b'reducefn', Fn2Tuple(self.server.reducefn))

        if self.server.collectfn:
            self.send_command(b'collectfn', Fn2Tuple(self.server.collectfn))
        self._start_new_task()
    
    def _map_done(self, command, data):
        self.server.taskmanager.map_done(data)
        self._start_new_task()

    def _reduce_done(self, command, data):
        self.server.taskmanager.reduce_done(data)
        self._start_new_task()

    def _map_fail(self, command, data):
        self.server.taskmanager.map_done(data)
        self.server.fail = True
        self._start_new_task()

    def _reduce_fail(self, command, data):
        self.server.taskmanager.reduce_done(data)
        self.server.fail = True
        self._start_new_task()

    def _start_new_task(self):
        command, data = self.server.taskmanager.next_task(self)
        if command == None:
            return
        self.send_command(command, data)

# **************************************************

class TaskManager(object):

    MAPPING    = 0
    MAPWAIT    = 1
    REDUCING   = 2
    FINISHED   = 3

    def __init__(self, datasource, server):

        self.datasource      = datasource
        self.server          = server
        self.logger          = server.logger 

        self.state           = TaskManager.MAPPING
        self.map_iter        = iter(self.datasource)
        self.working_maps    = {}
        self.map_results     = {}
        self.working_reduces = {}
        self.results         = {}

    def next_task(self, channel):

        if self.state == TaskManager.MAPPING:

            try:

                map_key = next(self.map_iter)
                if type(map_key) == tuple:
                    map_item = map_key
                else:
                    map_item = map_key, self.datasource[map_key]
                self.working_maps[map_item[0]] = map_item[1]
                return (b'map', map_item)

            except StopIteration:
                self.state = TaskManager.MAPWAIT
                self.queue = []

        if self.state == TaskManager.MAPWAIT:

            if len(self.working_maps) > 0:
                self.queue.append(channel)
                return None, None

            self.state = TaskManager.REDUCING
            self.reduce_iter = iter(self.map_results.items())

        if self.state == TaskManager.REDUCING:

            try:

                try:
                    while 1:
                        channel = self.queue.pop()
                        reduce_item = next(self.reduce_iter)
                        self.working_reduces[reduce_item[0]] = reduce_item[1]
                        channel.send_command(b'reduce', reduce_item)
                except IndexError:
                    pass

                reduce_item = next(self.reduce_iter)
                self.working_reduces[reduce_item[0]] = reduce_item[1]
                return (b'reduce', reduce_item)

            except StopIteration:
                self.state = TaskManager.FINISHED

        if self.state == TaskManager.FINISHED:
            if len(self.working_reduces) == 0:
                channel.last=True
            return (b'disconnect', None)

    def map_done(self, data):

        for (key, values) in data[1].items():
            if key not in self.map_results:
                self.map_results[key] = []
            self.map_results[key].extend(values)

        del self.working_maps[data[0]]
                                
    def reduce_done(self, data):

        self.results[data[0]] = data[1]
        del self.working_reduces[data[0]]

# **************************************************

if __name__ == '__main__':
    
    p = argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('server', 
                   help='Server Address')

    p.add_argument('-p', '--password',
                   help='Authentication password')

    p.add_argument('-P', '--port',
                   help='Server Port')

    p.add_argument('-t', '--timeout',
                   help='Server connect timeout',
                   type=float)

    p.add_argument('-n', '--nclients',
                   help='Number of clients to run',
                   type=int,
                   default=None)

    loggerValues = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    p.add_argument('-l', '--loglevel',
                   help='Logger level (debug, info, warning, error, critical)',
                   type=lambda x : x.upper(),
                   choices=loggerValues,
                   default='info')

    opt = p.parse_args()

    log = Logger(level=opt.loglevel,
                 fmt=FMT_LONG,
                 name='Clients')
                     
    log.info("Starting {}".format(sys.argv[0]))
 
    # Show options

    log.info(__doc__.splitlines()[1])
    log.info("Options specified:")
    for o in sorted(opt.__dict__.keys()):
        log.info("{:12s} = {}".format(o, opt.__dict__[o]))

    # Run the clients
    clients = Clients(logger=log, **vars(opt))

    rc = clients()
    if not rc:
        clients.join()

    log.info('{} ended with return code {}'.format(sys.argv[0], rc))
    sys.exit(rc)
