import json
import time
import os
import re
import argparse
from google import genai
from google.genai import types

# -----------------------------
# Configuration
# -----------------------------
# Default values - can be overridden by command line arguments
API_KEY = ""  # Will be set from CLI or environment
MODEL_NAME = "gemini-2.0-flash"
TOTAL_LINES = 50000
BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_DELAY = 2

# Temperature schedule: each value corresponds to one full round through all topics/subtopics
temperature_schedule = [0.0, 0.02, 0.04, 0.06, 0.08,
 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28,
 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48,
 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68,
 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88,
 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]


# Enhanced system prompt with more specific instructions
system_prompt = """You are a data generator tasked with creating high-quality synthetic training data for fine-tuning an AI language model. The training data must focus exclusively on one of the following four topics: Prompt Engineering, Foundational Models, Agentic AI, or Responsible AI.

For each topic, you will generate instruction-response pairs based on the subtopic specified by the user. Your goal is to create diverse, realistic, and useful training examples.

IMPORTANT REQUIREMENTS:
1. Each example MUST be a valid JSON object with the following keys:
   - "instruction": A clear, specific question or request related to the subtopic
   - "context": (Optional) Background information that helps frame the instruction
   - "response": A detailed, accurate, and helpful response to the instruction

2. DIVERSITY REQUIREMENTS:
   - Vary the instruction formats (questions, commands, requests for explanation)
   - Mix complexity levels (beginner to advanced queries)
   - Include different response types (explanations, steps, analyses, comparisons)
   - Vary the length of both instructions and responses

3. CONTENT QUALITY:
   - Ensure factual accuracy and technical correctness
   - Provide substantive, detailed responses that would be genuinely helpful
   - Avoid vague, generic, or repetitive content
   - Include specific examples, use cases, or scenarios where appropriate

4. OUTPUT FORMAT:
   - Each JSON object must be on a single line (no line breaks within objects)
   - There must be no markdown formatting, code blocks, or additional text
   - Ensure all JSON is properly escaped (quotes, backslashes, etc.)
   - Include no blank lines between JSON objects

Example of the expected format:
{"instruction": "Explain how to structure a prompt for a text-to-image model to get consistent artistic styles.", "context": "I'm working with Stable Diffusion for a design project.", "response": "When structuring prompts for text-to-image models like Stable Diffusion to achieve consistent artistic styles, follow these key principles: 1) Start with a clear art style reference (e.g., 'in the style of Monet' or 'cyberpunk aesthetic')..."}

Your goal is to generate exactly BATCH_SIZE examples for the given subtopic, ensuring each example is unique, valuable for training, and properly formatted.

DO NOT include any explanatory text, comments, or descriptions outside of the JSON objects themselves."""

# Define topics and subtopics in a fixed order
topic_order = ["Prompt Engineering", "Foundational Models", "Agentic AI", "Responsible AI"]

topics = {
    "Prompt Engineering": [
        "Effective Prompt Structuring",
        "Prompt Tuning Techniques",
        "Zero-shot Prompting",
        "Few-shot Prompting",
        "Chain-of-thought Prompting",
        "Prompting for Creativity",
        "Debugging Prompts",
        "Contextual Prompts",
        "Domain-specific Prompts",
        "Evaluating Prompt Performance"
    ],
    "Foundational Models": [
        "Model Architecture and Design",
        "Pretraining vs. Finetuning",
        "Scaling Laws",
        "Model Generalization",
        "Transfer Learning",
        "Data Curation for Foundational Models",
        "Ethical Considerations in Model Training",
        "Performance Benchmarks",
        "Model Robustness and Safety",
        "Multimodal Capabilities"
    ],
    "Agentic AI": [
        "Definition of Agentic Behavior",
        "Autonomous Decision-Making",
        "Self-improving AI Systems",
        "Ethical Implications of Agentic AI",
        "Agent-based Simulations",
        "Multi-agent Collaboration",
        "Goal-oriented Agentic Systems",
        "Risk Management in Agentic AI",
        "Adaptive Agentic Models",
        "Applications in Robotics"
    ],
    "Responsible AI": [
        "Fairness and Bias Mitigation",
        "Transparency and Explainability",
        "Data Privacy and Security",
        "Accountability in AI Systems",
        "Regulatory Compliance",
        "Ethical AI Guidelines",
        "AI for Social Good",
        "Human-AI Collaboration",
        "Adversarial Robustness",
        "Long-term Societal Impact"
    ]
}

# -----------------------------
# Utility Functions
# -----------------------------
def minimal_clean(text):
    """
    Perform minimal cleaning - only handle obvious markdown code blocks.
    This preserves more of the original content for manual fixing later.
    """
    # If the content is wrapped in markdown code blocks, extract just the content
    if text.strip().startswith("```") and text.strip().endswith("```"):
        # Find the first and last occurrences of ```
        start_idx = text.find("```") + 3
        # Find the end of the first line which might contain language specifier
        newline_idx = text.find("\n", start_idx)
        if newline_idx != -1:
            start_idx = newline_idx + 1
        end_idx = text.rfind("```")
        return text[start_idx:end_idx].strip()
    return text.strip()

def validate_and_fix_json(line, log_invalid=True):
    """
    Validates a line as JSON and attempts to fix common issues.
    Returns the fixed JSON string if successful, None otherwise.
    """
    # Skip empty lines
    if not line.strip():
        return None
    
    # First attempt: try to parse as-is
    try:
        json_obj = json.loads(line)
        # Validate required fields
        if all(key in json_obj for key in ['instruction', 'response']):
            return json.dumps(json_obj)  # Return normalized JSON
        else:
            if log_invalid:
                print(f"Missing required fields: {line[:100]}...")
            return None
    except json.JSONDecodeError:
        # Try some common fixes
        fixed_line = line
        
        # Fix 1: Remove trailing commas before closing braces
        fixed_line = re.sub(r',\s*}', '}', fixed_line)
        
        # Fix 2: Add missing quotes around keys
        fixed_line = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_line)
        
        # Fix 3: Escape unescaped quotes within string values
        # This is more complex and might require a more sophisticated approach
        
        # Try parsing again after fixes
        try:
            json_obj = json.loads(fixed_line)
            if all(key in json_obj for key in ['instruction', 'response']):
                if log_invalid:
                    print(f"Fixed JSON: {line[:50]}... -> {fixed_line[:50]}...")
                return json.dumps(json_obj)  # Return normalized JSON
            else:
                if log_invalid:
                    print(f"Fixed but missing required fields: {fixed_line[:100]}...")
                return None
        except json.JSONDecodeError:
            if log_invalid:
                print(f"Failed to fix JSON: {line[:100]}...")
            return None

def calculate_diversity_metrics(examples, n=5):
    """Calculate diversity metrics for a set of examples."""
    # Extract text for analysis
    instructions = [ex.get('instruction', '') for ex in examples if isinstance(ex, dict)]
    responses = [ex.get('response', '') for ex in examples if isinstance(ex, dict)]
    
    if not instructions or not responses:
        return {'error': 'No valid examples found'}
    
    # Calculate n-gram overlap for instructions
    instruction_ngrams = {}
    for instr in instructions:
        words = instr.lower().split()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            instruction_ngrams[ngram] = instruction_ngrams.get(ngram, 0) + 1
    
    # Calculate metrics
    repeated_ngrams = sum(1 for count in instruction_ngrams.values() if count > 1)
    total_ngrams = len(instruction_ngrams)
    
    # Calculate average instruction and response length
    avg_instruction_length = sum(len(instr.split()) for instr in instructions) / len(instructions)
    avg_response_length = sum(len(resp.split()) for resp in responses) / len(responses)
    
    return {
        'total_examples': len(examples),
        'unique_ngram_percentage': (1 - (repeated_ngrams / total_ngrams)) * 100 if total_ngrams > 0 else 0,
        'avg_instruction_length': avg_instruction_length,
        'avg_response_length': avg_response_length,
        'repeated_ngram_count': repeated_ngrams,
        'total_ngram_count': total_ngrams
    }

# -----------------------------
# Checkpointing Functions
# -----------------------------
def load_checkpoint(checkpoint_file):
    """Load progress from checkpoint file if it exists."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint: Generated {checkpoint['total_generated']} examples so far")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Default initial checkpoint
    return {
        'total_generated': 0,
        'lines_written': 0,
        'progress': {topic: {subtopic: 0 for subtopic in topics[topic]} for topic in topic_order},
        'temperature_index': 0
    }

def save_checkpoint(checkpoint_file, checkpoint_data):
    """Save current progress to checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Saved checkpoint: Generated {checkpoint_data['total_generated']} examples so far")

# -----------------------------
# API Handling
# -----------------------------
def call_gemini_api(system_prompt, user_prompt, desired_temp, max_tokens=8192):
    """
    Enhanced Gemini API caller with better error handling and rate limiting.
    """
    full_user_prompt = f"{user_prompt}\n[Simulated temperature: {desired_temp}]"
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=desired_temp
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            client = genai.Client(api_key=API_KEY)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[full_user_prompt],
                config=config
            )
            
            # Save raw response for debugging
            raw_response = response.text
            print(f"Raw response starts with: {raw_response[:100]}...")
            
            # Apply minimal cleaning
            cleaned = minimal_clean(raw_response)
            if cleaned:
                return cleaned
            else:
                return raw_response
                
        except Exception as e:
            error_message = str(e).lower()
            
            # Check for rate limit errors specifically
            if "rate limit" in error_message or "quota" in error_message:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            elif "internal server error" in error_message:
                # For server errors, wait a bit but not as long
                print(f"Server error. Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise Exception(f"Failed after {MAX_RETRIES} attempts: {e}")

# -----------------------------
# Processing Functions
# -----------------------------
def post_process_dataset(input_file, output_file):
    """
    Post-processes the generated dataset to ensure all lines are valid JSON.
    """
    valid_count = 0
    invalid_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(f"{output_file}.invalid.log", 'w', encoding='utf-8') as invalid_log:
        
        print(f"Post-processing dataset from {input_file} to {output_file}...")
        
        # Process line by line
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            # Try to validate and fix the JSON
            fixed_json = validate_and_fix_json(line, log_invalid=False)
            
            if fixed_json:
                # Check if this is a proper example with instruction and response
                try:
                    json_obj = json.loads(fixed_json)
                    # Perform additional quality checks
                    if len(json_obj.get('instruction', '')) < 10:
                        invalid_log.write(f"Line {line_num}: Instruction too short\n{line}\n\n")
                        invalid_count += 1
                        continue
                        
                    if len(json_obj.get('response', '')) < 50:
                        invalid_log.write(f"Line {line_num}: Response too short\n{line}\n\n")
                        invalid_count += 1
                        continue
                    
                    # Write the valid JSON to the output file
                    outfile.write(fixed_json + '\n')
                    valid_count += 1
                    
                    # Print progress
                    if valid_count % 100 == 0:
                        print(f"Processed {valid_count} valid examples...")
                        
                except Exception as e:
                    invalid_log.write(f"Line {line_num}: Error: {e}\n{line}\n\n")
                    invalid_count += 1
            else:
                invalid_log.write(f"Line {line_num}: Invalid JSON\n{line}\n\n")
                invalid_count += 1
    
    print(f"Post-processing complete.")
    print(f"Valid examples: {valid_count}")
    print(f"Invalid examples: {invalid_count}")
    print(f"Invalid examples logged to {output_file}.invalid.log")
    
    return valid_count, invalid_count

# -----------------------------
# Main Generation Function
# -----------------------------
def generate_dataset(output_filename, target_lines=50000):
    """Enhanced function to generate a high-quality synthetic dataset."""
    total_generated = 0
    lines_written = 0
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
    
    # Create files for raw output, logs, and checkpoint
    raw_output_file = output_filename
    log_filename = output_filename + ".log"
    checkpoint_file = output_filename + ".checkpoint.json"
    
    # Load checkpoint if it exists
    checkpoint = load_checkpoint(checkpoint_file)
    total_generated = checkpoint['total_generated']
    lines_written = checkpoint['lines_written']
    progress = checkpoint['progress']
    temp_index = checkpoint['temperature_index']
    
    with open(raw_output_file, "a" if total_generated > 0 else "w", encoding="utf-8") as outfile, \
         open(log_filename, "a" if total_generated > 0 else "w", encoding="utf-8") as logfile:
        
        # Calculate batches needed
        total_subtopics = sum(len(subtopics) for topic in topic_order for subtopics in [topics[topic]])
        
        # Calculate lines per complete cycle through all topics
        lines_per_round = total_subtopics * BATCH_SIZE
        required_rounds = (target_lines + lines_per_round - 1) // lines_per_round  # Ceiling division
        
        print(f"Each complete round generates approximately {lines_per_round} lines")
        print(f"Will run up to {required_rounds} complete rounds")
        print(f"Target total lines: {target_lines}")
        print(f"Starting from: {total_generated} examples, temp index {temp_index}")
        
        # Loop through temperatures
        while total_generated < target_lines and temp_index < len(temperature_schedule):
            temperature = temperature_schedule[temp_index]
            print(f"\n--- Starting round with temperature {temperature} ---\n")
            
            # Loop through each topic in the specified order
            for topic in topic_order:
                if total_generated >= target_lines:
                    break
                    
                # Loop through each subtopic for this topic
                for subtopic in topics[topic]:
                    # Skip if we've already reached our target
                    if total_generated >= target_lines:
                        break
                        
                    # Formulate a user prompt that includes the topic and subtopic
                    examples_generated = progress.get(topic, {}).get(subtopic, 0)
                    user_prompt = (
                        f"Generate {BATCH_SIZE} high-quality instruction-response pairs for "
                        f"{topic} - {subtopic} (Generated so far: {examples_generated})"
                    )
                    
                    try:
                        print(f"Generating data for '{topic} - {subtopic}'")
                        generated_text = call_gemini_api(system_prompt, user_prompt, temperature)
                        
                        # Log the raw response
                        logfile.write(f"--- {user_prompt} ---\n")
                        logfile.write(generated_text)
                        logfile.write("\n\n" + "="*80 + "\n\n")
                        logfile.flush()
                        
                        # Process the generated text to extract valid JSON lines
                        valid_lines = []
                        for line in generated_text.split('\n'):
                            if line.strip():
                                # Basic validation - we'll do thorough validation in post-processing
                                if line.strip().startswith('{') and line.strip().endswith('}'):
                                    valid_lines.append(line)
                        
                        # Write valid lines to the file
                        if valid_lines:
                            for line in valid_lines:
                                outfile.write(line + '\n')
                            outfile.flush()
                            
                            # Update progress
                            lines_added = len(valid_lines)
                            lines_written += lines_added
                            batch_examples = min(BATCH_SIZE, lines_added)  # Assume at most BATCH_SIZE valid examples
                            progress[topic][subtopic] = progress.get(topic, {}).get(subtopic, 0) + batch_examples
                            total_generated += batch_examples
                            
                            # Save checkpoint
                            checkpoint_data = {
                                'total_generated': total_generated,
                                'lines_written': lines_written,
                                'progress': progress,
                                'temperature_index': temp_index
                            }
                            save_checkpoint(checkpoint_file, checkpoint_data)
                            
                            # Calculate and display time estimates
                            elapsed_time = time.time() - start_time
                            examples_per_second = total_generated / elapsed_time if elapsed_time > 0 else 0
                            estimated_remaining = (target_lines - total_generated) / examples_per_second if examples_per_second > 0 else 0
                            
                            print(f"Added {lines_added} lines. Running total: ~{lines_written} lines")
                            print(f"Generated {total_generated}/{target_lines} examples")
                            print(f"Rate: {examples_per_second:.2f} examples/sec, Est. remaining: {estimated_remaining/60:.1f} minutes")
                        else:
                            print("Warning: No valid JSON lines found in response.")
                    except Exception as e:
                        print(f"Error generating for '{topic} - {subtopic}': {e}")
                        logfile.write(f"ERROR in '{topic} - {subtopic}': {e}\n\n")
                    
                    # Sleep to avoid rate limits
                    time.sleep(20)
            
            # Move to the next temperature
            temp_index += 1
            checkpoint_data = {
                'total_generated': total_generated,
                'lines_written': lines_written,
                'progress': progress,
                'temperature_index': temp_index
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
    
    # Post-process the raw output
    print("\nRaw generation complete. Starting post-processing...")
    valid_count, invalid_count = post_process_dataset(raw_output_file, output_filename)
    
    # Final stats
    elapsed_time = (time.time() - start_time) / 60  # minutes
    print(f"\nDataset generation and processing complete in {elapsed_time:.2f} minutes.")
    print(f"Final dataset contains {valid_count} valid examples in {output_filename}")
    print(f"Raw output saved to {raw_output_file}")
    print(f"Process logs saved to {log_filename}")
    
    return valid_count

# -----------------------------
# Command Line Interface
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic AI training dataset")
    parser.add_argument("--output", default="flash.jsonl", help="Output filename")
    parser.add_argument("--target", type=int, default=50000, help="Target number of examples")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size per API call")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--api-key", help="API key (overrides env variable)")
    args = parser.parse_args()
    
    # Set global variables from arguments
    global BATCH_SIZE, MODEL_NAME, API_KEY, TOTAL_LINES
    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model
    TOTAL_LINES = args.target
    
    # Set API key from arguments or environment
    if args.api_key:
        API_KEY = args.api_key
    elif os.environ.get("GEMINI_API_KEY"):
        API_KEY = os.environ.get("GEMINI_API_KEY")
    else:
        raise ValueError("API key must be provided via --api-key or GEMINI_API_KEY environment variable")
    
    # Run the generator
    generate_dataset(args.output, args.target)

if __name__ == "__main__":
    main()
