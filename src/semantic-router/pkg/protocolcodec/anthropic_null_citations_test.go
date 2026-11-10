/*
Copyright 2026 The vLLM Semantic Router Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicNullCitationsAcceptedInRequest(t *testing.T) {
	body := []byte(`{"model":"claude-3-5-sonnet","max_tokens":64,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"hello","citations":null}]}]}`)
	_, _, _, err := (AnthropicMessagesCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeRequest failed on citations:null: %v", err)
	}
}

func TestAnthropicNullCitationsAcceptedInResponse(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet","content":[{"type":"text","text":"hello","citations":null}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	_, _, _, err := (AnthropicMessagesCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeResponse failed on citations:null: %v", err)
	}
}

func TestAnthropicNonNullCitationsRejectedInRequest(t *testing.T) {
	body := []byte(`{"model":"claude-3-5-sonnet","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"text","text":"hello","citations":[{"type":"char_location","cited_text":"xyz"}]}]}]}`)
	_, _, _, err := (AnthropicMessagesCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_citations")
}

func TestAnthropicNonNullCitationsRejectedInResponse(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet","content":[{"type":"text","text":"hello","citations":[{"type":"char_location","cited_text":"xyz"}]}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	_, _, _, err := (AnthropicMessagesCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_citations")
}
