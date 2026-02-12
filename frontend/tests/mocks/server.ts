/**
 * MSW server for Node.js test environment
 *
 * This server intercepts outgoing HTTP requests during tests.
 */

import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);
