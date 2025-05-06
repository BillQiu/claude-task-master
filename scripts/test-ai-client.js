/**
 * test-ai-client.js
 * A simple test script to verify the AI unified client functionality
 */

import {
	createAIClient,
	invokeAIModel,
	determineAIProvider,
	AI_PROVIDER
} from './modules/ai-unified-client.js';

import { log } from './modules/utils.js';

// Test the client
async function testAIClient() {
	try {
		// Detect which provider we should use based on environment variables
		const provider = determineAIProvider();
		console.log(`Detected AI provider: ${provider}`);

		// Create the client
		const client = createAIClient();
		console.log('Successfully created AI client.');

		// Test a simple message with the client
		console.log('Sending a test message to the AI model...');
		const response = await invokeAIModel("Hello, what's your name?", {
			temperature: 0.7
		});

		console.log('\nResponse from AI model:');
		console.log('--------------------------------------------------');
		console.log(response);
		console.log('--------------------------------------------------');

		console.log('\nTest completed successfully.');
	} catch (error) {
		console.error('Test failed with error:', error.message);
		console.error('Stack trace:', error.stack);
		process.exit(1);
	}
}

// Run the test
testAIClient();
