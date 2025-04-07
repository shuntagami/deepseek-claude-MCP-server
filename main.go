package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	deepseekAPIBase = "https://api.deepseek.com"
	jsonRPCVersion  = "2.0"
)

// DeepseekMessage represents a message in the Deepseek API format
type DeepseekMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// DeepseekRequestPayload is the request structure for Deepseek API
type DeepseekRequestPayload struct {
	Model     string            `json:"model"`
	Messages  []DeepseekMessage `json:"messages"`
	Streaming bool              `json:"streaming"`
	MaxTokens int               `json:"max_tokens"`
}

// DeepseekResponseChunk represents a chunk of streaming response
type DeepseekResponseChunk struct {
	Choices []struct {
		Delta struct {
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
	} `json:"choices"`
}

// MCPRequest represents the incoming request structure for MCP
type MCPRequest struct {
	JSONRPC string                 `json:"jsonrpc"`
	ID      json.RawMessage        `json:"id"` // Use RawMessage to handle both string and number
	Method  string                 `json:"method"`
	Params  map[string]interface{} `json:"params,omitempty"`
}

// MCPResponse represents the outgoing response structure for MCP
type MCPResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  interface{}     `json:"result,omitempty"`
	Error   *MCPError       `json:"error,omitempty"`
}

// MCPError represents an error in the MCP protocol
type MCPError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// FastMCP represents a simplified version of FastMCP server
type FastMCP struct {
	Name string
}

// getDeepseekReasoning fetches reasoning from the DeepSeek API
func getDeepseekReasoning(query string) (string, error) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		// Try to use default key if environment variable not set
		apiKey = "enter your api key"
		log.Println("Using default API key (for development only)")
	}

	// Create request payload
	payload := DeepseekRequestPayload{
		Model:     "deepseek-reasoner",
		Messages:  []DeepseekMessage{{Role: "user", Content: query}},
		Streaming: true,
		MaxTokens: 2048,
	}

	// Marshal payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("error marshaling request payload: %w", err)
	}

	// Create request
	url := deepseekAPIBase + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// Create HTTP client with longer timeout
	client := &http.Client{
		Timeout: 300 * time.Second, // 5 minutes timeout
	}

	// Send request
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API returned non-200 status: %d, body: %s", resp.StatusCode, string(body))
	}

	// Process streaming response
	var reasoningData []string
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", fmt.Errorf("error reading response: %w", err)
		}

		// Check if the line starts with "data: "
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			data = strings.TrimSpace(data)

			// Check if this is the end of the stream
			if data == "DONE" {
				continue
			}

			// Try to parse the JSON data
			var chunk DeepseekResponseChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue // Skip this chunk if it can't be parsed
			}

			// Extract the reasoning content if available
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.ReasoningContent != "" {
				reasoningData = append(reasoningData, chunk.Choices[0].Delta.ReasoningContent)
			}
		}
	}

	return strings.Join(reasoningData, " "), nil
}

// handleInitialize processes the initialize request
func (mcp *FastMCP) handleInitialize(ctx context.Context, id json.RawMessage, params map[string]interface{}) MCPResponse {
	// Define tools for the capabilities
	tools := map[string]interface{}{
		"reason": map[string]interface{}{
			"description": "Process a query using DeepSeek's R1 reasoning engine",
			"inputSchema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"context": map[string]interface{}{
						"type":        "string",
						"description": "Optional background information for the query",
					},
					"question": map[string]interface{}{
						"type":        "string",
						"description": "The specific question to be analyzed",
					},
				},
				"required": []string{"question"},
			},
		},
	}

	// Return server information and capabilities
	result := map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"serverInfo": map[string]string{
			"name":    mcp.Name,
			"version": "1.0.0",
		},
		"capabilities": map[string]interface{}{
			"tools": tools,
		},
	}

	return MCPResponse{
		JSONRPC: jsonRPCVersion,
		ID:      id,
		Result:  result,
	}
}

// handleToolsList processes the tools/list request
func (mcp *FastMCP) handleToolsList(ctx context.Context, id json.RawMessage) MCPResponse {
	// Define the reason tool schema
	reasonTool := map[string]interface{}{
		"name":        "reason",
		"description": "Process a query using DeepSeek's R1 reasoning engine",
		"inputSchema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"context": map[string]interface{}{
					"type":        "string",
					"description": "Optional background information for the query",
				},
				"question": map[string]interface{}{
					"type":        "string",
					"description": "The specific question to be analyzed",
				},
			},
			"required": []string{"question"},
		},
	}

	return MCPResponse{
		JSONRPC: jsonRPCVersion,
		ID:      id,
		Result:  map[string]interface{}{"tools": []interface{}{reasonTool}},
	}
}

// handleResourcesList processes the resources/list request
func (mcp *FastMCP) handleResourcesList(ctx context.Context, id json.RawMessage) MCPResponse {
	return MCPResponse{
		JSONRPC: jsonRPCVersion,
		ID:      id,
		Result:  map[string]interface{}{"resources": []interface{}{}},
	}
}

// handlePromptsList processes the prompts/list request
func (mcp *FastMCP) handlePromptsList(ctx context.Context, id json.RawMessage) MCPResponse {
	return MCPResponse{
		JSONRPC: jsonRPCVersion,
		ID:      id,
		Result:  map[string]interface{}{"prompts": []interface{}{}},
	}
}

// handleReason processes a reason command and returns the response
func (mcp *FastMCP) handleReason(ctx context.Context, id json.RawMessage, params map[string]interface{}) MCPResponse {
	// Extract query parameters
	var context, question string
	if ctx, ok := params["context"].(string); ok {
		context = ctx
	}
	if q, ok := params["question"].(string); ok {
		question = q
	} else {
		return MCPResponse{
			JSONRPC: jsonRPCVersion,
			ID:      id,
			Error: &MCPError{
				Code:    -32602,
				Message: "Invalid params: missing required 'question' field",
			},
		}
	}

	// Format the query
	var fullQuery string
	if context != "" {
		fullQuery = fmt.Sprintf("%s\n%s", context, question)
	} else {
		fullQuery = question
	}

	// Get reasoning from DeepSeek
	reasoning, err := getDeepseekReasoning(fullQuery)
	if err != nil {
		log.Printf("Error getting DeepSeek reasoning: %v", err)
		return MCPResponse{
			JSONRPC: jsonRPCVersion,
			ID:      id,
			Error: &MCPError{
				Code:    -32603,
				Message: fmt.Sprintf("Internal error: %s", err.Error()),
			},
		}
	}

	// Format the result
	return MCPResponse{
		JSONRPC: jsonRPCVersion,
		ID:      id,
		Result:  map[string]interface{}{"result": fmt.Sprintf("<ant_thinking>\n%s\n</ant_thinking>\n\nNow we should provide our final answer based on the above thinking.", reasoning)},
	}
}

// handleToolsCall processes a tools/call request
func (mcp *FastMCP) handleToolsCall(ctx context.Context, id json.RawMessage, params map[string]interface{}) MCPResponse {
	// Extract tool name and arguments
	toolName, ok := params["name"].(string)
	if !ok {
		return MCPResponse{
			JSONRPC: jsonRPCVersion,
			ID:      id,
			Error: &MCPError{
				Code:    -32602,
				Message: "Invalid params: missing required 'name' field",
			},
		}
	}

	args, ok := params["arguments"].(map[string]interface{})
	if !ok {
		return MCPResponse{
			JSONRPC: jsonRPCVersion,
			ID:      id,
			Error: &MCPError{
				Code:    -32602,
				Message: "Invalid params: missing or invalid 'arguments' field",
			},
		}
	}

	// Check which tool to call
	if toolName == "reason" {
		// Extract query parameters
		var context, question string
		if ctx, ok := args["context"].(string); ok {
			context = ctx
		}
		if q, ok := args["question"].(string); ok {
			question = q
		} else {
			return MCPResponse{
				JSONRPC: jsonRPCVersion,
				ID:      id,
				Error: &MCPError{
					Code:    -32602,
					Message: "Invalid params: missing required 'question' field",
				},
			}
		}

		// Format the query
		var fullQuery string
		if context != "" {
			fullQuery = fmt.Sprintf("%s\n%s", context, question)
		} else {
			fullQuery = question
		}

		// Get reasoning from DeepSeek
		reasoning, err := getDeepseekReasoning(fullQuery)
		if err != nil {
			log.Printf("Error getting DeepSeek reasoning: %v", err)
			return MCPResponse{
				JSONRPC: jsonRPCVersion,
				ID:      id,
				Error: &MCPError{
					Code:    -32603,
					Message: fmt.Sprintf("Internal error: %s", err.Error()),
				},
			}
		}

		// Format the result
		return MCPResponse{
			JSONRPC: jsonRPCVersion,
			ID:      id,
			Result: map[string]interface{}{
				"content": []map[string]interface{}{
					{
						"type": "text",
						"text": fmt.Sprintf("<ant_thinking>\n%s\n</ant_thinking>\n\nNow we should provide our final answer based on the above thinking.", reasoning),
					},
				},
			},
		}
	}

	return MCPResponse{
		JSONRPC: jsonRPCVersion,
		ID:      id,
		Error: &MCPError{
			Code:    -32601,
			Message: fmt.Sprintf("Tool not found: %s", toolName),
		},
	}
}

// run starts the FastMCP server using stdio transport
func (mcp *FastMCP) run() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // Increase buffer size

	for scanner.Scan() {
		line := scanner.Text()
		log.Printf("Received: %s", line)

		// Parse the incoming JSON request
		var request MCPRequest
		if err := json.Unmarshal([]byte(line), &request); err != nil {
			log.Printf("Error parsing request: %v", err)
			// For parse errors, we can't properly include the request ID
			errorResp := MCPResponse{
				JSONRPC: jsonRPCVersion,
				Error: &MCPError{
					Code:    -32700,
					Message: fmt.Sprintf("Parse error: %s", err.Error()),
				},
			}
			respBytes, _ := json.Marshal(errorResp)
			fmt.Println(string(respBytes))
			continue
		}

		// Create context with longer timeout
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second) // 5 minutes timeout
		defer cancel()

		var response MCPResponse

		// Process the request based on method
		switch request.Method {
		case "initialize":
			response = mcp.handleInitialize(ctx, request.ID, request.Params)
		case "notifications/initialized", "notifications/cancelled":
			// Just acknowledge for notifications (no response needed)
			continue
		case "tools/list":
			response = mcp.handleToolsList(ctx, request.ID)
		case "resources/list":
			response = mcp.handleResourcesList(ctx, request.ID)
		case "prompts/list":
			response = mcp.handlePromptsList(ctx, request.ID)
		case "tools/call":
			response = mcp.handleToolsCall(ctx, request.ID, request.Params)
		case "reason":
			response = mcp.handleReason(ctx, request.ID, request.Params)
		default:
			response = MCPResponse{
				JSONRPC: jsonRPCVersion,
				ID:      request.ID,
				Error: &MCPError{
					Code:    -32601,
					Message: fmt.Sprintf("Method not found: %s", request.Method),
				},
			}
		}

		// Send the response (unless it's a notification that doesn't need a response)
		respBytes, err := json.Marshal(response)
		if err != nil {
			log.Printf("Error marshaling response: %v", err)
			errorResp := MCPResponse{
				JSONRPC: jsonRPCVersion,
				ID:      request.ID,
				Error: &MCPError{
					Code:    -32603,
					Message: "Internal error: could not marshal response",
				},
			}
			respBytes, _ = json.Marshal(errorResp)
		}

		log.Printf("Sending response: %s", string(respBytes))
		fmt.Println(string(respBytes))
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading from stdin: %v", err)
	}
}

func main() {
	// Set up logging to file
	logFile, err := os.OpenFile("mcp_server.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		// Set up multi-writer to log to both file and stderr
		multi := io.MultiWriter(os.Stderr, logFile)
		log.SetOutput(multi)
		defer logFile.Close()
	} else {
		log.Println("Warning: Could not open log file, using stderr only")
	}

	// Log startup information
	log.Println("Using environment variables from system")

	mcp := &FastMCP{
		Name: "deepseek-reasoner-claude",
	}
	log.Println("Starting deepseek-reasoner-claude MCP server...")
	mcp.run()
}
