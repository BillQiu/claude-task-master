/**
 * ai-unified-client.js
 * Unified client for AI services (Anthropic and OpenAI compatible API)
 */

import { Anthropic } from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import { log } from './utils.js';
import {
	invokeOpenAICompatibleChat,
	convertToOpenAIMessages,
	handleOpenAICompatibleError
} from './openai-compatible-client.js';

// Load environment variables
dotenv.config();

/**
 * Enum for supported AI provider types
 * @readonly
 * @enum {string}
 */
export const AI_PROVIDER = {
	ANTHROPIC: 'anthropic',
	OPENAI_COMPATIBLE: 'openai_compatible'
};

/**
 * Determine which AI provider to use based on environment
 * @returns {string} Provider type from AI_PROVIDER enum
 */
export function determineAIProvider() {
	// If OpenAI compatible API base URL is set, prefer it over Anthropic
	if (process.env.OPENAI_API_BASE_URL) {
		log(
			'info',
			`Using OpenAI compatible API at ${process.env.OPENAI_API_BASE_URL}`
		);
		return AI_PROVIDER.OPENAI_COMPATIBLE;
	}

	// If Anthropic API key is available, use Anthropic
	if (process.env.ANTHROPIC_API_KEY) {
		log('info', 'Using Anthropic API');
		return AI_PROVIDER.ANTHROPIC;
	}

	// Default to OpenAI compatible if no Anthropic key (will fail if no base URL either)
	log(
		'warn',
		'No AI provider configuration found. Defaulting to OpenAI compatible API.'
	);
	return AI_PROVIDER.OPENAI_COMPATIBLE;
}

/**
 * Create an Anthropic client
 * @returns {Anthropic} Configured Anthropic client
 * @throws {Error} If ANTHROPIC_API_KEY is missing
 */
export function createAnthropicClient() {
	if (!process.env.ANTHROPIC_API_KEY) {
		throw new Error(
			'ANTHROPIC_API_KEY environment variable is required for using Anthropic API.'
		);
	}

	return new Anthropic({
		apiKey: process.env.ANTHROPIC_API_KEY,
		// Add beta header for 128k token output
		defaultHeaders: {
			'anthropic-beta': 'output-128k-2025-02-19'
		}
	});
}

/**
 * Unified function to invoke AI for text completion
 * @param {Array|Object|string} messages - Messages in either Anthropic or OpenAI format
 * @param {Object} options - Additional options for the API call
 * @param {string} options.model - The model to use (default: from env var MODEL)
 * @param {number} options.maxTokens - Maximum number of tokens to generate (default: from env var MAX_TOKENS)
 * @param {number} options.temperature - Temperature for generation (default: from env var TEMPERATURE)
 * @param {string} options.provider - Explicitly specify provider (optional)
 * @returns {Promise<string>} The generated text response
 * @throws {Error} If the API call fails
 */
export async function invokeAIModel(messages, options = {}) {
	// Determine provider to use (explicit option, cached decision, or detect from env)
	const provider =
		options.provider || process.env.AI_PROVIDER || determineAIProvider();

	try {
		if (provider === AI_PROVIDER.OPENAI_COMPATIBLE) {
			// Convert messages to OpenAI format if needed
			const openAIMessages = convertToOpenAIMessages(messages);

			// Call OpenAI compatible API
			return await invokeOpenAICompatibleChat(openAIMessages, options);
		} else if (provider === AI_PROVIDER.ANTHROPIC) {
			// Create Anthropic client
			const client = createAnthropicClient();

			// Handle different message formats for Anthropic
			let promptMessages;
			let systemPrompt = '';

			if (typeof messages === 'string') {
				// Single string message
				promptMessages = messages;
			} else if (Array.isArray(messages)) {
				if (messages.length > 0 && messages[0].role && messages[0].content) {
					// OpenAI format, extract system message if present
					const systemMsg = messages.find((m) => m.role === 'system');
					if (systemMsg) {
						systemPrompt = systemMsg.content;
					}

					// Convert remaining messages to Anthropic format
					promptMessages = messages
						.filter((m) => m.role !== 'system')
						.map((m) => ({
							role: m.role === 'user' ? 'human' : 'assistant',
							content: m.content
						}));
				} else {
					// Simple array of strings, convert to Anthropic format
					promptMessages = messages;
				}
			} else if (messages.system || messages.messages) {
				// Already in Anthropic format
				systemPrompt = messages.system || '';
				promptMessages = messages.messages;
			}

			// Call Anthropic API
			const response = await client.messages.create({
				model:
					options.model || process.env.MODEL || 'claude-3-7-sonnet-20250219',
				max_tokens:
					options.maxTokens || parseInt(process.env.MAX_TOKENS || '64000'),
				temperature:
					options.temperature || parseFloat(process.env.TEMPERATURE || '0.2'),
				system: systemPrompt,
				messages: promptMessages
			});

			return response.content[0].text;
		} else {
			throw new Error(`Unsupported AI provider: ${provider}`);
		}
	} catch (error) {
		// Handle provider-specific errors
		if (provider === AI_PROVIDER.OPENAI_COMPATIBLE) {
			const errorMessage = handleOpenAICompatibleError(error);
			log('error', errorMessage);
			throw new Error(errorMessage);
		} else {
			// Handle Anthropic errors
			const errorMessage = `Anthropic API error: ${error.message}`;
			log('error', errorMessage);
			throw new Error(errorMessage);
		}
	}
}

/**
 * Create a suitable AI client based on environment configuration
 * @param {Object} options - Options for client creation
 * @param {string} options.provider - Explicitly specify provider (optional)
 * @returns {Object} AI client object appropriate for the selected provider
 * @throws {Error} If client creation fails
 */
export function createAIClient(options = {}) {
	const provider =
		options.provider || process.env.AI_PROVIDER || determineAIProvider();

	try {
		if (provider === AI_PROVIDER.OPENAI_COMPATIBLE) {
			// For OpenAI compatible, we don't return a client object directly
			// Instead, return an adapter object that mimics the Anthropic client interface
			return {
				provider: AI_PROVIDER.OPENAI_COMPATIBLE,
				messages: {
					create: async (params) => {
						// Convert to OpenAI format
						const messages = [];

						// Add system message if present
						if (params.system) {
							messages.push({
								role: 'system',
								content: params.system
							});
						}

						// Add other messages
						if (Array.isArray(params.messages)) {
							params.messages.forEach((msg) => {
								messages.push({
									role: msg.role === 'human' ? 'user' : 'assistant',
									content: msg.content
								});
							});
						} else if (typeof params.messages === 'string') {
							messages.push({
								role: 'user',
								content: params.messages
							});
						}

						// Call OpenAI compatible API
						const content = await invokeOpenAICompatibleChat(messages, {
							model: params.model,
							maxTokens: params.max_tokens,
							temperature: params.temperature
						});

						// Return in a format similar to Anthropic's response
						return {
							id: 'openai-compatible-message',
							content: [{ text: content, type: 'text' }],
							model: params.model,
							role: 'assistant',
							usage: { input_tokens: 0, output_tokens: 0 }
						};
					}
				}
			};
		} else if (provider === AI_PROVIDER.ANTHROPIC) {
			// Return actual Anthropic client
			return createAnthropicClient();
		} else {
			throw new Error(`Unsupported AI provider: ${provider}`);
		}
	} catch (error) {
		const errorMessage = `Failed to create AI client: ${error.message}`;
		log('error', errorMessage);
		throw new Error(errorMessage);
	}
}
