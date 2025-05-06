/**
 * openai-compatible-client.js
 * OpenAI compatible API client for Task Master
 */

import axios from 'axios';
import dotenv from 'dotenv';
import { log } from './utils.js';

// Load environment variables
dotenv.config();

/**
 * Get configuration for the OpenAI-compatible API client
 * @returns {Object} Configuration object with baseURL and apiKey
 * @throws {Error} If required configuration is missing
 */
export function getOpenAICompatibleConfig() {
  const baseURL = process.env.OPENAI_API_BASE_URL;
  if (!baseURL) {
    throw new Error('OPENAI_API_BASE_URL environment variable is missing. Set it to use OpenAI compatible API (e.g., http://localhost:11434/v1).');
  }

  // API key may be optional for some local API servers (e.g., Ollama),
  // so we don't throw an error if it's missing
  const apiKey = process.env.OPENAI_API_KEY || 'not-needed';

  return { baseURL, apiKey };
}

/**
 * Create an axios client configured for OpenAI compatible API
 * @returns {Object} Configured axios client
 * @throws {Error} If required configuration is missing
 */
export function createOpenAICompatibleClient() {
  const { baseURL, apiKey } = getOpenAICompatibleConfig();
  
  // Create and return a configured axios instance
  return axios.create({
    baseURL,
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey && apiKey !== 'not-needed' ? { 'Authorization': `Bearer ${apiKey}` } : {})
    }
  });
}

/**
 * Invokes an OpenAI compatible API for chat completion
 * @param {Array} messages - Array of message objects in OpenAI format (role, content)
 * @param {Object} options - Additional options for the API call
 * @param {string} options.model - The model to use (default: from env var MODEL)
 * @param {number} options.maxTokens - Maximum number of tokens to generate (default: from env var MAX_TOKENS)
 * @param {number} options.temperature - Temperature for generation (default: from env var TEMPERATURE)
 * @returns {Promise<string>} The generated text response
 * @throws {Error} If the API call fails
 */
export async function invokeOpenAICompatibleChat(messages, options = {}) {
  try {
    const client = createOpenAICompatibleClient();
    
    // Prepare request payload with defaults from environment variables
    const payload = {
      model: options.model || process.env.MODEL,
      messages,
      max_tokens: options.maxTokens || parseInt(process.env.MAX_TOKENS || '64000'),
      temperature: options.temperature || parseFloat(process.env.TEMPERATURE || '0.2')
    };

    log('debug', `Calling OpenAI compatible API with model: ${payload.model}`);
    
    // Make the API request
    const response = await client.post('/chat/completions', payload);
    
    // Extract and return the response text
    if (response.data && 
        response.data.choices && 
        response.data.choices.length > 0 && 
        response.data.choices[0].message && 
        response.data.choices[0].message.content) {
      return response.data.choices[0].message.content.trim();
    } else {
      throw new Error('Unexpected API response format');
    }
  } catch (error) {
    // Handle different types of errors
    if (error.response) {
      // The server responded with a status code outside the 2xx range
      const status = error.response.status;
      const data = error.response.data;
      throw new Error(`API error (${status}): ${JSON.stringify(data)}`);
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error(`Network error: No response received. Check your OPENAI_API_BASE_URL (${process.env.OPENAI_API_BASE_URL}).`);
    } else {
      // Something else happened while setting up the request
      throw new Error(`Error setting up request: ${error.message}`);
    }
  }
}

/**
 * Convert Anthropic-style messages to OpenAI format
 * @param {Array|Object} messages - Messages in Anthropic format or a single system/user message
 * @returns {Array} Messages in OpenAI format
 */
export function convertToOpenAIMessages(messages) {
  // Handle single message (string or object)
  if (typeof messages === 'string') {
    return [{ role: 'user', content: messages }];
  }
  
  // If it's already in OpenAI format (array of {role, content} objects), return as is
  if (Array.isArray(messages)) {
    if (messages.length > 0 && messages[0].role && messages[0].content) {
      return messages;
    }
    
    // Convert array of strings (alternating user/assistant) to OpenAI format
    return messages.map((message, index) => ({
      role: index % 2 === 0 ? 'user' : 'assistant',
      content: message
    }));
  }
  
  // If it's a single message object with role/content, wrap in array
  if (messages.role && messages.content) {
    return [messages];
  }
  
  // Handle Anthropic-specific format (system, messages)
  if (messages.system && messages.messages) {
    const result = [];
    
    // Add system message if present
    if (messages.system) {
      result.push({
        role: 'system',
        content: messages.system
      });
    }
    
    // Add other messages
    if (Array.isArray(messages.messages)) {
      messages.messages.forEach(msg => {
        if (msg.role && msg.content) {
          // Map Anthropic roles to OpenAI roles
          const role = msg.role === 'human' ? 'user' : 
                      msg.role === 'assistant' ? 'assistant' : 
                      msg.role;
          result.push({
            role,
            content: msg.content
          });
        }
      });
    }
    
    return result;
  }
  
  // Default to treating as user message
  return [{ role: 'user', content: JSON.stringify(messages) }];
}

/**
 * Handle errors from OpenAI compatible API
 * @param {Error} error - The error from API
 * @returns {string} User-friendly error message
 */
export function handleOpenAICompatibleError(error) {
  // Check for network/connection errors
  if (error.message?.includes('Network error')) {
    return `Connection error: Unable to reach the AI service at ${process.env.OPENAI_API_BASE_URL}. Please check that your server is running and the URL is correct.`;
  }
  
  // Check for authentication errors
  if (error.message?.includes('401') || error.message?.includes('403')) {
    return 'Authentication error: The API key provided is invalid or missing. Please check your OPENAI_API_KEY.';
  }
  
  // Check for model errors
  if (error.message?.includes('model')) {
    return `Model error: The requested model '${process.env.MODEL}' may not be available. Please check that you have the correct model name.`;
  }
  
  // Default error message
  return `Error communicating with AI service: ${error.message}`;
} 