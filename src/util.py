# util.py

import os
import json

from ExtractionUtils import *

#
#   Helpers
#
def find_codeblocks(input_str):
    # Find all codeblocks in the input string
    codeblocks = []
    lines = input_str.split("\n")
    current_codeblock = []
    active = False

    for idx, line in enumerate(lines):
        if line.startswith("```"):
            if (active == True):
                # Finish off the current codeblock
                codeblocks.append(current_codeblock)
                current_codeblock = []
                active = False
            else:
                # Start a new codeblock
                active = True
        else:
            # If we're currently in the middle of a codeblock, add the line to the current codeblock (we never want the ``` to be included in the codeblock)
            if (active == True):
                current_codeblock.append(line)

    # Note: turn each of the lines in the codeblock into a single string, so we can parse it as JSON later
    for idx in range(len(codeblocks)):
        codeblocks[idx] = "\n".join(codeblocks[idx])

    return codeblocks


# From CodeScientist
# This is a general function for trimming -- use the logging-specific one for logs.
def trimPromptComponent(componentIn:str, maxTokens:int=10000):
    lines = componentIn.split('\n')
    # If it's below the max tokens, return it
    token_count = countTokens(componentIn)
    if (token_count <= maxTokens):
        return componentIn

    # If it's above the max tokens, try to trim it
    # Try to trim the lines from the middle out
    windowSize = 1
    trimmedComponent = ""
    while (token_count > maxTokens) and (windowSize < len(lines)):
        # Trim out the middle lines, +/- windowSize
        trimmedLines = []
        middleIdx = len(lines) // 2
        trimStartIdx = max(0, middleIdx - windowSize)
        trimEndIdx = min(len(lines), middleIdx + windowSize)
        trimmedLines = lines[:trimStartIdx] + ["# (Up to " + str(windowSize) + " lines trimmed for space)"] + lines[trimEndIdx:]
        trimmedComponent = "\n".join(trimmedLines)
        token_count = countTokens(trimmedComponent)

        # Increase the window size
        windowSize += 1

    if (token_count > maxTokens):
        print("WARNING: Could not trim the component to fit within the token limit.")
        return "# WARNING: Could not trim the component to fit within the token limit.  This may mean single lines exceed the token limit.\n" + componentIn
    # Return the trimmed component
    return trimmedComponent


# From CodeScientist
# This is a function for trimming the log file, that (tries to) keeps all things marked as 'error', since those are likely high-importance.
def trimPromptComponentLog(logIn:list, maxTokens:int=10000):
    print("Trimming log with " + str(len(logIn)) + " lines to fit within " + str(maxTokens) + " tokens.")
    # Each element in the log is a dictionary with two keys: `type` and `message`.  We want to, at a minimum, keep all `error` types.
    # First, check if the log is already below the token limit

    # Ensure that 'logIn' is a list
    if (type(logIn) != list):
        print("##########################################")
        print("ERROR: 'logIn' is not a list.  Returning.")
        print("Type is: " + str(type(logIn)))
        print("##########################################")
        return ["ERROR: component provided to `trimPromptComponentLog` is not a list (type is " + str(type(logIn)) + ")"]

    token_count = countTokens(json.dumps(logIn, indent=4))
    if (token_count <= maxTokens):
        return json.dumps(logIn, indent=4)
    initial_token_count = token_count

    # Pre-compute the token counts for each line -- this is a slow step, so if we pre-compute it, it speeds everything up.
    #tokenCountsPerLine = []
    tokenCountsPerLine = [0] * len(logIn)
    for idx in range(len(logIn)):
        tokenCountsPerLine[idx] = countTokens(json.dumps(logIn[idx], indent=4))

    messageJSON = [{"type": "meta", "message": "# (Up to " + str(1234567890) + " lines trimmed for space, but messages with type `error` retained)"}]
    token_count_message = countTokens(json.dumps(messageJSON, indent=4))

    # If it's above the max tokens, try to trim it
    # Try to trim the lines from the middle out
    windowSize = 1
    trimmedLog = []
    iterations = 0
    windowSizeStep = 1
    if (len(logIn) > 10000):
        windowSizeStep = len(logIn) // 1000

    while (token_count > maxTokens) and (windowSize < len(logIn)):
        iterations += 1
        #print("Iteration: " + str(iterations))

        # Trim out the middle lines, +/- windowSize
        trimmedLog = []

        # Find the boundaries of the part of the middle that we'll trim out
        middleIdx = len(logIn) // 2
        trimStartIdx = max(0, middleIdx - windowSize)
        trimEndIdx = min(len(logIn), middleIdx + windowSize)

        # But, we have to keep the errors
        middleErrors = []
        middleErrorTokenCounts = []
        token_count_middle_errors = 0
        for middleIdx1 in range(trimStartIdx, trimEndIdx):
            logEntry = logIn[middleIdx1]
            try:
                if (logEntry["type"].lower() == "error"):
                    middleErrors.append(logEntry)
                    middleErrorTokenCounts.append(tokenCountsPerLine[middleIdx1])
                    token_count_middle_errors += tokenCountsPerLine[middleIdx1]
            except:
                pass
        # If the number of tokens in the middle errors ends up being more than 25% of the message, then start subsampling the middle errors (just include the ones at the start and end)
        if (token_count_middle_errors > 0.25 * maxTokens):
            #print("*** Token count of middle errors is more than 25% of the message.  Subsampling the middle errors.")

            # Try to ratchet it up a bit
            middleWindowSize = 1
            found = False
            for middleWindowSize in range(1, len(middleErrors)//2):
                candidateMiddleErrors = middleErrors[:middleWindowSize] + middleErrors[-middleWindowSize:]
                # Use middleErrorTokenCounts
                token_count_candidate_middle_errors = sum(middleErrorTokenCounts[:middleWindowSize]) + sum(middleErrorTokenCounts[-middleWindowSize:])
                if (token_count_candidate_middle_errors > 0.25 * maxTokens):
                    found = True
                    break

            #print("Window size: " + str(middleWindowSize))

            # If we still can't get it down, then just take the first and last 5
            if (found == True):
                middleErrors = middleErrors[:middleWindowSize] + middleErrors[-middleWindowSize:]
                token_count_candidate_middle_errors = sum(middleErrorTokenCounts[:middleWindowSize]) + sum(middleErrorTokenCounts[-middleWindowSize:])

            else:
                min_middle_errors = 5
                if (len(middleErrors) < 2 * min_middle_errors):
                    min_middle_errors = len(middleErrors) // 2

                middleErrors = middleErrors[:min_middle_errors] + middleErrors[-min_middle_errors:]
                token_count_candidate_middle_errors = sum(middleErrorTokenCounts[:middleWindowSize]) + sum(middleErrorTokenCounts[-middleWindowSize:])

            token_count_middle_errors = token_count_candidate_middle_errors

        # Add the token counts from the first half and second half, plus any that we have to keep in the middle (that are 'errors')
        token_count_first_half = sum(tokenCountsPerLine[:trimStartIdx])
        token_count_second_half = sum(tokenCountsPerLine[trimEndIdx:])

        token_count_estimate = token_count_first_half + token_count_message + token_count_middle_errors + token_count_second_half

        # Count the number of lines that are included
        numLinesIncluded = trimStartIdx + (len(logIn) - trimEndIdx)

        MINIMUM_LINES_TO_INCLUDE = 20       # If we have less than this many lines, stop trimming the log.
        if (token_count_estimate > maxTokens) and (numLinesIncluded > 20):
            # If the estimate is too high, then increase the window size
            #windowSize += 1
            windowSize += windowSizeStep        # Makes it faster for large log files
            continue

        #print("Window: 0 -> " + str(trimStartIdx) + " and " + str(trimEndIdx) + " -> " + str(len(logIn)) + " (including " + str(len(middleErrors)) + " middle errors)")
        #print("Token count estimate: " + str(token_count_estimate) + " (first_half: " + str(token_count_first_half) + ", message: " + str(token_count_message) + ", middle_errors: " + str(token_count_middle_errors) + ", second_half: " + str(token_count_second_half) + ")")

        # If we reach here, then the estimate is below the max tokens, so we can assemble the trimmed log
        trimmedLogMiddle = [{"type": "meta", "message": "# (Up to " + str(windowSize) + " lines trimmed for space, but messages with type `error` retained)"}]
        trimmedLogMiddle2 = [{"type": "meta", "message": "# (End of trimming)"}]
        trimmedLog = logIn[:trimStartIdx] + trimmedLogMiddle + middleErrors + trimmedLogMiddle2 + logIn[trimEndIdx:]

        # End here
        break

    # Check the final token count
    token_count = countTokens(json.dumps(trimmedLog, indent=4))
    print("Final token count: " + str(token_count) + " . (Max tokens: " + str(maxTokens) + ", initial_token_count = " + str(initial_token_count) + ")")

    # If it's still above the max tokens, then return an error
    TOKEN_COUNT_TOLERANCE = 1000
    if (token_count > (maxTokens + TOKEN_COUNT_TOLERANCE)):
        print("WARNING: Could not trim the log to fit within the token limit.  Trying back-off method.")
        # Use a backoff method -- just start adding in lines (alternating from the top/bottom) until we're within the token limit
        trimmedLog = []
        logTop = []
        logBottom = []
        topIdx = 0
        bottomIdx = len(logIn) - 1
        middleMessage = [{"type": "meta", "message": "# (This is only a partial log -- lines trimmed for space"}]

        cycleCount = 0
        while (token_count < (maxTokens + TOKEN_COUNT_TOLERANCE)) and (cycleCount < 1000):     # Add a hard limit of 1000 cycles here, for the backoff, just in case
            # Check to see if adding these lines will put us over the token limit
            logTop = logIn[:topIdx]
            logBottom = logIn[bottomIdx:]
            tempTrimmedLog = middleMessage + logTop + middleMessage + logBottom
            token_count = countTokens(json.dumps(tempTrimmedLog, indent=4))
            if (token_count < (maxTokens + TOKEN_COUNT_TOLERANCE)):
                trimmedLog = tempTrimmedLog

            # Increment the top and bottom indices
            topIdx += 1
            bottomIdx -= 1

            # If the top and bottom have met, then break
            if (topIdx >= bottomIdx):
                break

            cycleCount += 1
            #print("Cycle: " + str(cycleCount) + "  Token count: " + str(token_count) + "  Max tokens: " + str(maxTokens) + " topIdx: " + str(topIdx) + " bottomIdx: " + str(bottomIdx))

        #return "# WARNING: Could not trim the log to fit within the token limit.  This may mean single lines exceed the token limit.\n" + json.dumps(logIn, indent=4)

    # Return the trimmed log
    print("Trimmed log to " + str(len(trimmedLog)) + " lines.")
    return json.dumps(trimmedLog, indent=4)


# Fix JSON
def parse_and_fix_json(json_str, model_str="gpt-5-mini"):
    # Try to parse the JSON string
    try:
        json_obj = json.loads(json_str)
        return json_obj, 0.0
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

        prompt = "The following JSON had at least one parsing error. Can you please examine the JSON below, and fix any syntax errors (but do NOT change the content of the JSON in any way, except for fixing syntactic errors)?\n\n" + json_str
        prompt += "JSON:\n"
        prompt += "```\n"
        prompt += json_str + "\n"
        prompt += "```\n"
        prompt += "Parsing error: " + str(e) + "\n"
        prompt += "# Output Format\n"
        prompt += "You can think as much as you would like before outputing the final JSON, but please output the final (fixed, fully-correct and parsable) JSON between a set of codeblocks (```) as the final thing you output.\n"

        # Run prompt
        responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.0, jsonOut=True)
        return responseJSON, cost
