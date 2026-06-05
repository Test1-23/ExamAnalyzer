#!/usr/bin/env node
/**
 * Event Bus — hierarchical pub/sub for KP lifecycle coordination.
 *
 * Main bus (port 3030):
 *   POST  /publish/:type     — publish an event (JSON body)
 *   WS    /ws                — subscribe to events (send {subscribe: "type"})
 *   GET   /health            — health check + listener summary
 *
 * Sub-bus (port 3031–3035 per module):
 *   Same API, isolated event namespace per module.
 *
 * Startup: node server.js [--port 3030]
 */

const http = require('http');
const { WebSocketServer } = require('ws');
const { EventEmitter } = require('events');

// ============================================================
// EventBus class — isolated namespace
// ============================================================

class EventBus {
  constructor(name) {
    this.name = name;
    this.emitter = new EventEmitter();
    this.emitter.setMaxListeners(200);
    this.eventLog = [];       // ring buffer of recent events
    this.maxLogSize = 1000;
    this.subscriberCount = 0;
  }

  publish(type, payload) {
    const event = { type, payload, ts: new Date().toISOString() };
    this.eventLog.push(event);
    if (this.eventLog.length > this.maxLogSize) this.eventLog.shift();
    this.emitter.emit(type, payload);
    // Also emit to wildcard subscribers
    this.emitter.emit('*', event);
    return event;
  }

  subscribe(type, handler) {
    this.emitter.on(type, handler);
    this.subscriberCount++;
    return () => {
      this.emitter.off(type, handler);
      this.subscriberCount--;
    };
  }

  getStats() {
    return {
      name: this.name,
      events: this.eventLog.length,
      subscribers: this.subscriberCount,
      eventTypes: this.emitter.eventNames().filter(n => n !== '*'),
    };
  }
}

// ============================================================
// HTTP + WebSocket server factory
// ============================================================

function createServer(bus, port) {
  const server = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');

    if (req.method === 'OPTIONS') {
      res.writeHead(204).end();
      return;
    }

    // Health check
    if (req.method === 'GET' && req.url === '/health') {
      res.writeHead(200).end(JSON.stringify({ status: 'ok', ...bus.getStats() }));
      return;
    }

    // List recent events
    if (req.method === 'GET' && req.url === '/events') {
      const recent = bus.eventLog.slice(-50);
      res.writeHead(200).end(JSON.stringify(recent));
      return;
    }

    // Publish
    if (req.method === 'POST' && req.url.startsWith('/publish/')) {
      const eventType = req.url.split('/publish/')[1];
      if (!eventType) {
        res.writeHead(400).end(JSON.stringify({ error: 'missing event type' }));
        return;
      }
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => {
        try {
          const payload = JSON.parse(body);
          const event = bus.publish(eventType, payload);
          res.writeHead(200).end(JSON.stringify({ ok: true, ts: event.ts }));
        } catch (e) {
          res.writeHead(400).end(JSON.stringify({ error: 'invalid JSON: ' + e.message }));
        }
      });
      return;
    }

    res.writeHead(404).end(JSON.stringify({ error: 'not found' }));
  });

  // WebSocket for subscriptions
  const wss = new WebSocketServer({ server, path: '/ws' });

  wss.on('connection', ws => {
    const cleanups = [];

    ws.on('message', raw => {
      try {
        const msg = JSON.parse(raw.toString());

        if (msg.subscribe) {
          const type = msg.subscribe;
          const unsub = bus.subscribe(type, data => {
            if (ws.readyState === 1) { // WebSocket.OPEN
              ws.send(JSON.stringify({ type, data, ts: new Date().toISOString() }));
            }
          });
          cleanups.push(unsub);
          ws.send(JSON.stringify({ subscribed: type, bus: bus.name }));
        }

        if (msg.subscribe_all) {
          const unsub = bus.subscribe('*', event => {
            if (ws.readyState === 1) {
              ws.send(JSON.stringify(event));
            }
          });
          cleanups.push(unsub);
          ws.send(JSON.stringify({ subscribed: '*', bus: bus.name }));
        }
      } catch (e) {
        ws.send(JSON.stringify({ error: e.message }));
      }
    });

    ws.on('close', () => cleanups.forEach(fn => fn()));
  });

  server.listen(port, () => {
    console.log(`[${bus.name}] Event bus running on port ${port}`);
    console.log(`  Health:  http://127.0.0.1:${port}/health`);
    console.log(`  Publish: POST http://127.0.0.1:${port}/publish/:type`);
    console.log(`  Subscribe: ws://127.0.0.1:${port}/ws`);
  });

  return server;
}

// ============================================================
// Main — start main bus + sub-buses
// ============================================================

const mainPort = parseInt(process.env.EVENT_BUS_PORT || '3030', 10);

// Main bus
const mainBus = new EventBus('main');
createServer(mainBus, mainPort);

// Sub-buses for each module
const subBuses = {
  knowledge: new EventBus('knowledge'),
  analysis: new EventBus('analysis'),
  pipeline: new EventBus('pipeline'),
  chat: new EventBus('chat'),
  web: new EventBus('web'),
};

Object.entries(subBuses).forEach(([name, bus], i) => {
  createServer(bus, mainPort + 1 + i);
});

console.log(`\nAll buses started. Main: :${mainPort}, Sub: :${mainPort + 1}-:${mainPort + 5}`);
console.log('Press Ctrl+C to stop.\n');
