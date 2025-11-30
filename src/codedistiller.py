# codedistiller.py

import os
import json
import shutil
import random
import argparse
import traceback
from datetime import datetime

# enumeration
import enum

from util import *                  # Helpers
from ExtractionUtils import *

# Threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

#  Pygments for syntax highlighting
from pygments import highlight
from pygments.lexers import PythonLexer, JsonLexer
from pygments.formatters import HtmlFormatter

py_formatter = HtmlFormatter(nowrap=True)



# CodeDistiller version
VERSION_CODEDISTILLER = "1.0.0"

# Enumeration for file classes
class FileType(enum.Enum):
    # General types
    CODE_GENERAL = "code_general"
    DOCUMENTATION_GENERAL = "documentation_general"
    RUNSCRIPT = "runscript"
    DATA = "data"  # Data files, e.g., CSV, JSON, etc.
    OTHER = "other"

    # Specific types
    CODE_EXAMPLE = "code_example"                       # Example of how to use the code
    CODE_MAIN_ENTRY_POINT = "code_main_entry_point"     # The main entry point of the code

    DOCUMENTATION_RUNNING = "documentation_running"     # Documentation on how to run the code
    DOCUMENTATION_EXAMPLE = "documentation_example"     # Documentation of examples of how to use the code

    # ToString method
    def __str__(self):
        return self.value

    # Method for JSON serialization
    def to_json(self):
        return self.value


# Scratch directory for temporary files
SCRATCH_DIRECTORY = os.getenv("MAKE_EXAMPLE_SCRATCH", "scratch")
if not os.path.exists(SCRATCH_DIRECTORY):
    os.makedirs(SCRATCH_DIRECTORY)

# Output directory for output examples
# OUTPUT_DIRECTORY = os.getenv("MAKE_EXAMPLE_OUTPUT", "output")
# if not os.path.exists(OUTPUT_DIRECTORY):
#     os.makedirs(OUTPUT_DIRECTORY)




#
#   Class: MakeExampleFromRepo
#
class MakeExampleFromRepo():
    # Constructor
    # repo_name should be a full github repo name like "https://www.github.com/username/repo"
    def __init__(self, repo_url:str, max_runtime_mins:int=30, max_debug_iterations:int=6, output_directory:str="output"):
        self.repo_url = repo_url
        self.runtimes = {}
        self.costs = {}

        self.max_runtime_mins = max_runtime_mins                # Maximum runtime (in minutes) for the container, per debug iteration
        self.max_debug_iterations = max_debug_iterations        # Maximum number of debug iterations to run

        self.current_state = None

        self.status = "UNKNOWN" # Status of the example generation process
        self.status_message = ""

        # Try to create the output directory, if it doesn't already exist
        if (not os.path.exists(output_directory)):
            print("Creating output directory: " + str(output_directory))
            os.makedirs(output_directory)
        self.base_path = output_directory
        self.pathOut = None
        self.repo_path = None


    def get_total_cost(self):
        # Calculate the total cost of all operations
        total_cost = 0.0
        for key in self.costs.keys():
            total_cost += self.costs[key]
        return total_cost

    def get_total_runtime_secs(self):
        # Calculate the total runtime of all operations
        total_runtime = 0.0
        for key in self.runtimes.keys():
            total_runtime += self.runtimes[key]
        return total_runtime


    # The main entry point for making an example from the repository
    #def begin(self, model_str_code:str="o4-mini", model_str_fast:str="gpt-4.1-mini"):
    def begin(self, model_str_code:str="openai/gpt-5", model_str_fast:str="openai/gpt-5-mini"):
        try:
            return self.begin_(model_str_code=model_str_code, model_str_fast=model_str_fast)
        except Exception as e:
            print("Error: " + self.status_message)
            traceback.print_exc()
            return None
        finally:
            # Clean-up: Remove the cloned repository directory, if it exists
            if (self.repo_path is not None) and (os.path.exists(self.repo_path)):
                print("Cleaning up: Removing cloned repository directory: " + str(self.repo_path))
                shutil.rmtree(self.repo_path, ignore_errors=True)


    def begin_(self, model_str_code:str="openai/gpt-5", model_str_fast:str="openai/gpt-5-mini"):
        # Helper function for saving the current state
        def save_state(filename:str, state:dict):
            # Add the latest costs, times, and status to the state
            state["total_cost"] = self.get_total_cost()
            state["costs"] = self.costs
            state["total_runtime_secs"] = self.get_total_runtime_secs()
            state["runtimes"] = self.runtimes
            state["status"] = self.status
            state["status_message"] = self.status_message

            self.current_state = state  # Update the current state in the class

            # Save the current state to a file
            print("Writing: " + str(filename))
            with open(filename, 'w', encoding='utf-8') as f:
                # Use JSON serialization to handle enums
                json.dump(state, f, indent=4, ensure_ascii=False, default=lambda o: o.value if isinstance(o, enum.Enum) else o)

            return state


        # Step 0: Make an output directory (based on the repo username and repo name)
        repo_name = self.repo_url.split("/")[-1]
        repo_username = self.repo_url.split("/")[-2]
        combined_name = f"{repo_username}_{repo_name}"
        combined_name += datetime.now().strftime("_%Y%m%d_%H%M%S")  # Add a timestamp to the directory name, so we don't overwrite previous runs
        output_dir = os.path.join(self.base_path, combined_name)
        self.pathOut = output_dir  # Save the output directory path in the class
        # Create the output directory, if it doesn't already exist
        if not os.path.exists(self.pathOut):
            os.makedirs(self.pathOut)
        print("Output directory: " + self.pathOut)
        filenameOutState = os.path.join(self.pathOut, f"current_state.json")


        # Begin the process of making an example from the repository
        print(f"Beginning to make example from repository: {self.repo_url}")
        self.status = "STARTED"

        # Step 1: Clone the repository
        print("Cloning repository...")
        start_time_clone = time.time()
        repo_path = self.clone_repo(self.repo_url)
        self.repo_path = repo_path
        self.runtimes["clone_repo"] = time.time() - start_time_clone
        print(f"Cloned repository to: {repo_path}")

        # TODO: Check for failure (e.g. if the repository was not successfully cloned, for any reason)

        # Step 2: Classify the files in the repository
        print("Classifying files in the repository...")
        start_time_classify = time.time()
        classification_result = self.classify_repo_files(repo_path, model_str=model_str_fast, model_str_summary=model_str_code)
        self.runtimes["classify_repo_files"] = time.time() - start_time_classify
        print("File classification complete.")

        # Unpack response
        repository_description = classification_result.get("repository_description", None)
        classification_list = classification_result.get("classification_list", None)
        if (repository_description is None):
            self.status = "ERROR"
            self.status_message = "No repository description found."
            print("Error: No repository description found.")
            return None
        if (classification_list is None) or (len(classification_list) == 0):
            self.status = "ERROR"
            self.status_message = "No files classified in the repository."
            print("Error: No files classified in the repository.")
            return None

        # Step 3: Generate the first attempt at the code examples
        print("Generating initial code example...")
        start_time_generate = time.time()
        initial_code_result = self.generate_initial_example(repository_description=repository_description, repo_url=self.repo_url, selected_files=classification_list, model_str=model_str_code)
        self.runtimes["generate_initial_example"] = time.time() - start_time_generate
        print("Initial code example generation complete.")

        if (initial_code_result is None) or (isinstance(initial_code_result, dict) == False) or ("code" not in initial_code_result):
            self.status = "ERROR"
            self.status_message = "Failed to generate initial code example."
            print("Error: Failed to generate initial code example.")
            return None


        # Setup a data structure with the current state of the example
        current_state = {
            "repo_url": self.repo_url,
            "repository_description": repository_description,
            "status": self.status,
            "status_message": self.status_message,
            "model_str_code": model_str_code,
            "model_str_fast": model_str_fast,
            "total_cost": self.get_total_cost(),
            "costs": self.costs,
            "total_runtime_secs": self.get_total_runtime_secs(),
            "runtimes": self.runtimes,
            "max_runtime_mins": self.max_runtime_mins,
            "max_debug_iterations": self.max_debug_iterations,
            "reflection_idx": 0,
            "is_finished": False,
            "change_log": [],
            "code": initial_code_result["code"],
            "requirements": initial_code_result["requirements"],
            "runscript": initial_code_result["runscript"],
            "metadata": initial_code_result["metadata"],
            "error_information": None,
            "classification_list": classification_list,
            "execution_result": [],
        }

        # Step 5: Reflection
        self.costs["reflect_code_example"] = 0.0
        start_time_reflect = time.time()

        for reflection_idx in range(0, self.max_debug_iterations):
            current_state["reflection_idx"] = reflection_idx        # Store the reflection index

            # Step 4B: Run the initial code example
            print("Running code example...")
            start_time_run = time.time()
            # def run_code_example_in_container(self, code:str, requirements:str, runscript:str, metadata:dict, path_out:str, timeout_sec:int=60*15):
            initial_execution_result = self.run_code_example_in_container(
                code=current_state["code"],
                requirements=current_state["requirements"],
                runscript=current_state["runscript"],
                metadata=current_state["metadata"],
                path_out=self.pathOut,
                timeout_sec=self.max_runtime_mins * 60  # Convert minutes to seconds
            )
            if (initial_execution_result is None) or (isinstance(initial_execution_result, dict) == False):
                self.status = "ERROR"
                self.status_message = "Failed to execute code example."
                print("Error: Failed to run initial code example.")
                current_state = save_state(filenameOutState, current_state)  # Save the current state with the error
                return None

            self.runtimes["run_code_example_in_container"] = time.time() - start_time_run
            current_state["execution_result"].append(initial_execution_result)
            print("Initial code example run complete.")

            # Step 4C: Save the initial execution result to a file
            current_state = save_state(filenameOutState, current_state)

            print("-"*40)
            print(f"Reflection iteration {reflection_idx + 1} of {self.max_debug_iterations}...")
            print("-"*40)
            # Check if the current state is valid
            reflection_result = self.reflect_code_example(
                current_example_state=current_state,
                model_str=model_str_code,
            )
            if (reflection_result is None) or (isinstance(reflection_result, dict) == False):
                self.status = "ERROR"
                self.status_message = "Failed to reflect on code example."
                print("Error: Failed to reflect on code example.")
                # Update the current state with the error
                current_state["status"] = self.status
                current_state["status_message"] = self.status_message
                current_state = save_state(filenameOutState, current_state)
                return None

            # Check for an error condition (e.g. prompt too long)
            if ("error" in reflection_result) and (reflection_result["error"] is not None):
                self.status = "ERROR"
                self.status_message = "Error during reflection: " + str(reflection_result["error"])
                print("Error: " + self.status_message)
                # Update the current state with the error
                current_state["status"] = self.status
                current_state["status_message"] = self.status_message
                current_state = save_state(filenameOutState, current_state)
                return None


            self.runtimes["reflect_code_example"] = time.time() - start_time_reflect
            self.costs["reflect_code_example"] += reflection_result.get("total_cost", 0.0)

            # Update the current state with the reflection result
            # NOTE: If the reflection process determines the code is OK, then it will not return these values (code/requirements/etc.), so we'll need to retain the old values.
            if ("code" in reflection_result) and (reflection_result["code"] is not None):
                current_state["code"] = reflection_result["code"]
            if ("requirements" in reflection_result) and (reflection_result["requirements"] is not None):
                current_state["requirements"] = reflection_result["requirements"]
            if ("runscript" in reflection_result) and (reflection_result["runscript"] is not None):
                current_state["runscript"] = reflection_result["runscript"]
            if ("metadata" in reflection_result) and (reflection_result["metadata"] is not None):
                current_state["metadata"] = reflection_result["metadata"]
            if ("error_information" in reflection_result) and (reflection_result["error_information"] is not None):
                current_state["error_information"] = reflection_result["error_information"]


            # Get the next addition to the changelog
            changelog_entry = {}
            changelog_entry["iteration"] = reflection_idx + 1
            metadata = reflection_result.get("metadata", {})
            error_information = reflection_result.get("error_information", None)
            changelog_entry["error_information"] = error_information
            changelog_entry["issues_identified"] = metadata.get("issues_identified", None)
            changelog_entry["changes_made"] = metadata.get("changes_made", None)
            current_state["change_log"].append(changelog_entry)

            # Check to see if the reflection process has been marked as complete
            current_state["is_finished"] = error_information.get("is_finished", False)     # If this is TRUE, then the reflection process is complete.

            # Save the current state to a file (again)
            current_state = save_state(filenameOutState, current_state)

            # If the example is marked as successful, then we can stop the reflection process.
            if (current_state["is_finished"] == True):
                print("Reflection process marked as successful.  Stopping reflection iterations.")
                break

        # Determine the final status
        if (current_state["is_finished"] == True):
            self.status = "SUCCESS"
            self.status_message = "Example generation completed successfully."
        else:
            self.status = "FAILED"
            self.status_message = "Example generation did not complete successfully."

        current_state = save_state(filenameOutState, current_state)

        print("Example generation process complete.")
        print("Path: " + str(self.pathOut))
        print("Final result (success): " + str(current_state["is_finished"]))

        # Generate an HTML report from the final state
        filename_in = os.path.join(self.pathOut, "current_state.json")
        filename_out = os.path.join(self.pathOut, "report.html")
        filename_highlight_js = "highlight-js/highlight.min.js"
        filename_highlight_css = "highlight-js/github.min.css"
        convert_log_to_html_report(filename_in, filename_out, highlight_js_path=filename_highlight_js, highlight_css_path=filename_highlight_css)


    #
    #   Cloning/Repo Handling
    #

    # Clone the repository into a scratch directory
    def clone_repo(self, url:str):
        repo_name = url.split("/")[-1]
        repo_path = os.path.join(SCRATCH_DIRECTORY, repo_name)
        # Add a timestamp to the repo path to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_path += "_" + timestamp
        # Add a random 4 digit number to the repo path to avoid collisions
        rand_suffix = random.randint(1000, 9999)
        repo_path += "_" + str(rand_suffix)
        if not os.path.exists(repo_path):
            os.system(f"git clone {url} {repo_path}")
        return repo_path


    #
    #   Repository File Classification
    #

    def classify_repo_files(self, repo_path_local:str, MAX_FILES_PER_TYPE:int=10, MAX_WORKERS:int=20, model_str="gpt-4.1-mini", model_str_summary:str="o4-mini"):
        total_cost = 0.0

        # Step 1: Define the initial classification of files based on their extensions
        # This is a temporary classification that will be refined later.
        classification_initial = {
            # Documentation types
            "md": FileType.DOCUMENTATION_GENERAL,
            "txt": FileType.DOCUMENTATION_GENERAL,

            # Code types
            "py": FileType.CODE_GENERAL,
            "ipynb": FileType.CODE_GENERAL,  # Jupyter Notebook
            "js": FileType.CODE_GENERAL,
            "java": FileType.CODE_GENERAL,
            "cpp": FileType.CODE_GENERAL,
            "c": FileType.CODE_GENERAL,
            "go": FileType.CODE_GENERAL,
            "rs": FileType.CODE_GENERAL,  # Rust
            "ts": FileType.CODE_GENERAL,  # TypeScript
            "rb": FileType.CODE_GENERAL,  # Ruby
            "php": FileType.CODE_GENERAL,  # PHP

            # Runscripts/Shell scripts
            "sh": FileType.RUNSCRIPT,
            "bash": FileType.RUNSCRIPT,
            "bat": FileType.RUNSCRIPT,  # Windows batch files

            # Data files
            "json": FileType.DATA,
            "csv": FileType.DATA,
            "tsv": FileType.DATA,
            "xml": FileType.DATA,
            "yaml": FileType.DATA,
            "yml": FileType.DATA,
            "xls": FileType.DATA,
            "xlsx": FileType.DATA,
        }

        # Step 2: Crawl the repository to find all files (and, provide a temporary classification to them).
        filenames_and_classifications = []
        for root, dirs, files in os.walk(repo_path_local):
            for file in files:
                file_path_local = os.path.join(root, file)
                file_extension = file.split('.')[-1].lower() if '.' in file else ''
                file_type = classification_initial.get(file_extension, FileType.OTHER)
                # Also keep track of the 'nesting level' -- how many directories deep the file is.
                nesting_level = file_path_local.count(os.sep) - repo_path_local.count(os.sep)
                # Make a file path that's relative to the repository root
                file_path_relative = os.path.relpath(file_path_local, repo_path_local)
                file_name = os.path.basename(file_path_local)
                packed = {
                    "nesting_level": nesting_level,
                    "file_type": file_type,
                    "file_name": file_name,
                    "file_path_local": file_path_local,
                    "file_path_repo": file_path_relative,  # Relative path from the repository root
                }
                filenames_and_classifications.append(packed)

        # DEBUG: Show all the file classifications found in the repository
        print("Files found in the repository:")
        for item in filenames_and_classifications:
            #print(f" NL {item['nesting_level']}: {item['file_type'].value:25}: {item['file_path']}")
            print(json.dumps(item, default=lambda o: o.value if isinstance(o, enum.Enum) else o))

        # Step 3: Print a histogram of the file types found in the repository
        file_type_histogram = {}
        for item in filenames_and_classifications:
            file_type = item["file_type"]
            if file_type not in file_type_histogram:
                file_type_histogram[file_type] = 0
            file_type_histogram[file_type] += 1

        print("File type histogram:")
        # Sort alphabetically by key (file type)
        for file_type, count in sorted(file_type_histogram.items(), key=lambda x: x[0].value):
            print(f"{file_type.value}: {count}")


        # Step 4: For the first N files of each type (code, or documentation, or runscript), try to classify them further.
        # First, assemble a shortlist of files to classify further.
        # Sort the files by nesting level
        files_code = [item for item in filenames_and_classifications if item["file_type"] == FileType.CODE_GENERAL]
        files_documentation = [item for item in filenames_and_classifications if item["file_type"] == FileType.DOCUMENTATION_GENERAL]
        files_runscript = [item for item in filenames_and_classifications if item["file_type"] == FileType.RUNSCRIPT]
        # Sort by nesting level (ascending)
        files_code.sort(key=lambda x: x["nesting_level"])
        files_documentation.sort(key=lambda x: x["nesting_level"])
        files_runscript.sort(key=lambda x: x["nesting_level"])

        # Take the first MAX_FILES_PER_TYPE files of each type
        files_to_classify = {
            FileType.CODE_GENERAL: files_code[:MAX_FILES_PER_TYPE],
            FileType.DOCUMENTATION_GENERAL: files_documentation[:MAX_FILES_PER_TYPE],
            FileType.RUNSCRIPT: files_runscript[:MAX_FILES_PER_TYPE],
        }

        responses = []
        # Step 5A: Try to generate a high-level summary of the repository using the README file.
        # Find the README file (if it exists)
        #model_str_summary = "o3-mini"  # Model to use for summary generation
        #model_str_summary = "o4-mini"  # Model to use for summary generation
        repository_description = "This is a code repository, but did not contain a README file to generate a summary from."
        highly_relevant_files = []
        readme_file_path = os.path.join(repo_path_local, "README.md")
        if os.path.exists(readme_file_path):
            print(f"Found README file: {readme_file_path}")
            # Generate a summary of the repository using the README file
            response = self.generate_repo_summary_with_llm(file_path=readme_file_path, model_str=model_str_summary)
            try:
                repository_description = json.dumps(response["response"], indent=4)
                if ("highly_relevant_files" in response["response"]):
                    highly_relevant_files = response["response"]["highly_relevant_files"]
                print(f"Generated repository description: {repository_description}")
            except Exception as e:
                print(f"Error generating repository description: {e}")

            # Cost
            try:
                total_cost += response["total_cost"]
            except Exception as e:
                print(f"Error adding cost for README file: {e}")

            responses.append(response)

        # TODO: Should ensure that any files listed under 'highly_relevant_files' in the README file are included in the shortlist of files to classify further.
        # Get a list of all the files we have shortlisted to classify
        shortlisted_files = []
        for file in files_to_classify.values():
            for item in file:
                shortlisted_files.append(item["file_path_repo"])
        missing_files = []
        # Find any files that are in the highly_relevant_files list, but not in the shortlisted files
        for file in highly_relevant_files:
            if file not in shortlisted_files:
                # Find it in `filenames_and_classifications` and add it to `missing_files`
                for item in filenames_and_classifications:
                    if item["file_path_repo"] == file:
                        missing_files.append(item)
                        break
        missing_files.sort(key=lambda x: x["nesting_level"])  # Sort by nesting level (ascending)
        # Add the missing files to the shortlist
        #files_to_classify["MISSING_FILES"] = missing_files[:MAX_FILES_PER_TYPE]  # Limit to MAX_FILES_PER_TYPE
        files_to_classify[FileType.OTHER] = missing_files[:MAX_FILES_PER_TYPE]  # Add missing files under 'other' category, so it doesn't break the loop below

        print("Missing files:")
        print(f"Found {len(missing_files)} missing files that were listed in the README file as highly relevant, but not in the shortlist of files to classify further.")
        for idx, item in enumerate(missing_files):
            print("Missing File " + str(idx+1) + ": " + json.dumps(item, default=lambda o: o.value if isinstance(o, enum.Enum) else o))

        # Step 5B: For each file in the shortlist, try to classify it further using the LLM.
        #model_str_classification = "gpt-4.1-mini"  # Model to use for classification

        # # Single-thread version
        # for file_type, files in files_to_classify.items():
        #     print(f"Classifying {len(files)} files of type {file_type.value}...")
        #     for file_info in files:
        #         file_path = file_info["file_path_local"]
        #         print(f"Classifying file: {file_path}")
        #         # Call the LLM to classify the file
        #         response = self.classify_file_with_llm(repository_description=repository_description, file_path=file_path, model_str=model_str_classification)
        #         if response is not None:
        #             # Print the response
        #             print(f"Response for {file_path}: {json.dumps(response, indent=2)}")
        #         else:
        #             print(f"Failed to classify file: {file_path}")

        #         # Add the response to the list of responses
        #         responses.append(response)

        #         filenameOut = "responses.debug.json"
        #         with open(filenameOut, 'w', encoding='utf-8') as f:
        #             json.dump(responses, f, indent=4, ensure_ascii=False)

        #         try:
        #             # Add the cost of the response to the total cost
        #             total_cost += response["total_cost"]
        #         except Exception as e:
        #             print(f"Error adding cost for file {file_path}: {e}")

        #         print("Total running cost: " + str(total_cost))

        # Parallel version (using ThreadPoolExecutor)
        from tqdm import tqdm
        MAX_RUNTIME_SECS = 60*2
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a list to hold the futures
            futures = []
            for file_type, files in files_to_classify.items():
                print(f"Classifying {len(files)} files of type {file_type.value} in parallel...")
                for file_info in files:
                    file_path = file_info["file_path_local"]
                    print(f"Classifying file: {file_path}")
                    # Submit the classification task to the executor
                    future = executor.submit(self.classify_file_with_llm, repository_description=repository_description, file_path=file_path, model_str=model_str, extra_info_to_pack=file_info)
                    futures.append(future)
            # Wait for all futures to complete
            num_responses = 0
            for future in concurrent.futures.as_completed(futures, timeout=MAX_RUNTIME_SECS):
                num_responses += 1
                # update progress bar
                with tqdm(total=len(futures), desc="Classifying files", unit="file") as pbar:
                    pbar.update(num_responses)

                try:
                    response = future.result()
                    if response is not None:
                        # Print the response
                        response_str = json.dumps(response["response"], indent=2, ensure_ascii=False, default=lambda o: o.value if isinstance(o, enum.Enum) else o)
                        print(f"Response for {response['file_path']}: {response_str}")
                    else:
                        print("Failed to classify file.")
                    # Add the response to the list of responses
                    responses.append(response)

                    try:
                        # Add the cost of the response to the total cost
                        total_cost += response["total_cost"]
                        print("Total running cost: " + str(total_cost))
                    except Exception as e:
                        print(f"Error adding cost for file: {e}")

                    # Debug output
                    #filenameOut = "responses.debug.json"
                    #with open(filenameOut, 'w', encoding='utf-8') as f:
                    #    json.dump(responses, f, indent=4, ensure_ascii=False, default=lambda o: o.value if isinstance(o, enum.Enum) else o)


                except Exception as e:
                    import traceback
                    print(f"Error classifying file: {e}")
                    traceback.print_exc()

        # Filter out any responses that are None
        responses = [response for response in responses if response is not None]

        # Step 6: Sort the classification responses by category (code, documentation, runscript, other), and how relevant they are.
        # Debug output
        # filenameOut = "responses.debug.json"
        # with open(filenameOut, 'w', encoding='utf-8') as f:
        #    json.dump(responses, f, indent=4, ensure_ascii=False, default=lambda o: o.value if isinstance(o, enum.Enum) else o)

        # Save costs
        self.costs["classify_repo_files"] = total_cost

        # Return the sorted responses
        return {
            "repository_description": repository_description,
            "classification_list": responses,
        }




    # This function will attempt to further classify a given file using a language model.
    def classify_file_with_llm(self, repository_description:str, file_path:str, model_str:str="gpt-4.1-mini", extra_info_to_pack:dict=None):
        MAX_TOKENS_PER_FILE = 50000
        #MAX_TOKENS_PER_FILE = 100000

        # If the file was truncated, then `file_original_tokens` will be populated with the original number of tokens.
        def mkPrompt(repository_description:str, filename:str, file_contents:str, file_original_tokens:int=None, reflection:str=None):
            prompt = ""
            prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
            prompt += "\n"

            prompt += "# Task\n"
            prompt += "You are part of a system that is working to create short, complete examples of code repositories to demonstrate how they function to non-experts in a brief, informative, maximally useful way.\n"
            prompt += "As part of this task, you will be going through each file in the code repository, and classifying it into specific categories (code, documentation, etc.), as well as whether it (a) appears to be primary documentation, an existing example, or other high-value file that is directly useful for this task."
            prompt += "After going through the entire repository (one file at a time), the files that were identified to be the most highly relevant will be provided to a separate prompt to try to build a complete, working code example.  The success of that process will depend upon providing highly relevant (and only highly relevant) code, documentation, and/or runscripts as input.\n"
            prompt += "\n"

            prompt += "# Context\n"
            prompt += "This is a file from a code repository, where the overall repository has the following (automatically generated) description:\n"
            prompt += "```\n"
            prompt += repository_description + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# Query File\n"
            prompt += "Here is the file to investigate:\n"
            prompt += f"Filename: {filename}\n"
            if (file_original_tokens is not None):
                prompt += f"Note: Due to its size, this file was truncated to {MAX_TOKENS_PER_FILE} tokens, but it originally had {file_original_tokens} tokens.\n"
            prompt += "```\n"
            prompt += file_contents + "\n"
            prompt += "```\n"
            prompt += "\n"

            if (reflection is not None):
                prompt += "\n"
                prompt += "# Reflection\n"
                prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
                prompt += "```\n"
                prompt += reflection + "\n"
                prompt += "```\n"
                prompt += "\n"

            prompt += "# Why you are doing this\n"
            prompt += "You are part of an automated scientific discovery system that is working to rapidly accelerate the pace of scientific discovery, and generally make scientific discovery faster, less expensive, and more accessible.  Performing well at this task could have large positive benefits for humanity.  Performing poorly at this task is considered a critical failure, and will (at a minimum) cost time, money, and other resources, but could also lead to incorrect results (a strongly negative outcome).\n"
            prompt += "\n"

            prompt += "# File Classification\n"
            prompt += "You will classify this file into one of the following categories:\n"
            prompt += "- `code_example`: A code file that contains one or more example(s) of how to use the code, or a specific example of the code in action\n"
            prompt += "- `code_main_entry_point`: A code file that appears to be the main entry point of the code/repository, such as the primary script to run (e.g. for example, that takes command line parameters as input)\n"
            prompt += "- `code_general`: A code file that is not an example or main entry point.\n"
            prompt += "- `documentation_running`: A documentation file that contains instructions on how to run the code, how to set it up, etc.\n"
            prompt += "- `documentation_example`: A documentation file that contains examples of how to use the code, or how to run examples.\n"
            prompt += "- `documentation_general`: A documentation file that is not an example or instructions on how to run the code.\n"
            prompt += "- `runscript_example`: A file that is a script to run an example of the code, or to run the code in a specific way.\n"
            prompt += "- `runscript_running`: A file that is a script to run the code, configure the code, or otherwise to setup/run the code.\n"
            prompt += "- `runscript_general`: A file that is a script, but not an example or instructions on how to run the code.\n"
            prompt += "- `data`: A file that is a data file, such as a CSV, JSON, etc.\n"
            prompt += "- `other`: A file that does not fit into any of the above categories, or is not a code file, documentation file, or runscript.\n"
            prompt += "\n"

            prompt += "# Output Format\n"
            prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
            prompt += "\n"
            prompt += "Your JSON response must be a dictionary, that contains each of the following keys (NOTE: If no information is available for a key, you must still return the key, but with a value of `null` -- not a blank string):\n"
            prompt += "- `classification`(str): The file classification, from the above set of classifications.  The classification must be a string that is exactly identical to one of the classifications above. (must not be null)\n"
            prompt += "- `description`(str): A brief description (1-2 sentences) of the file, and its purpose in the repository. This should be a highly concise summary of the file's contents and purpose, particularly in the context of this task (setting up the repository and generating a running example of its usage) not a detailed description. (must not be null)\n"
            prompt += "- `important_information`(str): Does this file contain important information that is relevant to the task of setting up/running the code, and generating a code example?  If so, write that information here (not a description of the information, but the information itself) in a highly concise, but not lossy, way.  It will be directly passed on in the example generation process. (null otherwise)\n"
            prompt += "- `computational_requirements`(str): Does this file specify any requirements for running the code, such as GPUs, TPUs, multi-core CPUs, a large amount of disk, or other specialized requirements?  If so, list them here (briefly) with as much detail as possible. (null otherwise)\n"
            prompt += "- `highly_relevant_files`(list of str): Does this file reference any other files in the repository that it suggests are very highly relevant to the task of setting up/running the code, and generating a code example?  If so, list them here as a list of strings, with each string being the relative path to the file from the repository root.  If no such files are referenced, return an empty list.\n"
            prompt += "- `how_relevant_rating`(int): A rating of how absolutely critical this file is to the task of setting up the overall repository, and generating an example of the main purpose of the repository (1-5 scale). 5 = absolutely critical, unlikely to be possible without this file. 3 = potentially relevant, but not critical. 1 = not relevant at all. (must not be null)\n"
            prompt += "- `stand_alone_example`(bool): Does this file contain a complete, stand-alone example of how to use the code in the repository?  If so, return true.  If not, return false. (must not be null)\n"
            prompt += "\n"
            prompt += "For example:\n"
            prompt += "```\n"
            prompt += "{\n"
            prompt += "  \"classification\": \"documentation_running\",\n"
            prompt += "  \"description\": \"This file contains instructions on how to setup and run the main code in this repository, and perform the repository's primary function (xyz...)\",\n"
            prompt += "  \"important_information\": \"This file gives the following 4 step procedure for configuring a conda environment, setting up the configuration files, and running the code: 1. ... # and so on.\",\n"
            prompt += "  \"computational_requirements\": \"This code normally suggests minimum requirements of a GPU with at least 16GB of VRAM, and a multi-core CPU with at least 8 cores. It also suggests it could be run in a slower mode using CPU cores only, and at least 32GB of system RAM.\",\n"
            prompt += "  \"highly_relevant_files\": [\n"
            prompt += "    \"docs/setup.md\",\n"
            prompt += "    \"examples/example1.py\"\n"
            prompt += "  ],\n"
            prompt += "  \"how_relevant_rating\": 5\n"
            prompt += "}\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "# Important Notes\n"
            prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
            prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
            prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
            prompt += "- All your information must be accurate. Do not hallucinate.\n"

            return prompt



        # Step 1: Read the file contents
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_contents = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

        # Step 2: Count the number of tokens in the file
        file_contents_tokens = countTokens(file_contents)
        file_original_tokens = file_contents_tokens
        # If the file is too large, then truncate it
        is_truncated = False
        if (file_contents_tokens > MAX_TOKENS_PER_FILE):
            file_contents = trimToMaxTokens(file_contents, MAX_TOKENS_PER_FILE)
            is_truncated = True

        # Step 3: Make the prompt
        prompt = ""
        if (is_truncated == False):
            prompt = mkPrompt(repository_description=repository_description, filename=file_path, file_contents=file_contents, file_original_tokens=None)
        else:
            prompt = mkPrompt(repository_description=repository_description, filename=file_path, file_contents=file_contents, file_original_tokens=file_original_tokens)

        # Check maximum token count
        MAX_TOKENS_FULL_PROMPT = 100000 - 5000
        if ("gpt-oss-120b" in model_str):
            MAX_TOKENS_FULL_PROMPT = 80000 - 5000
        prompt_tokens = countTokens(prompt)
        if (prompt_tokens > MAX_TOKENS_FULL_PROMPT):
            print(f"ERROR: Prompt is too long ({prompt_tokens} tokens, max {MAX_TOKENS_FULL_PROMPT}).  Skipping file {file_path}.")
            return None

        # Step 4: Call the language model
        total_cost = 0.0
        responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.0, jsonOut=False)
        total_cost += cost

        # Step 5: Parse the response
        if (responseJSON is None):
            print(f"Error: No response from the language model for file {file_path}")
            # Retry with a slightly higher temperature
            responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.1, jsonOut=False)
            total_cost += cost

        # Add the file tokens to the response
        if (responseJSON is not None) and (isinstance(responseJSON, dict)):
            responseJSON["file_num_tokens"] = file_original_tokens

        # Return
        packed = {}
        if (extra_info_to_pack is not None):
            # Add the extra info to the packed response
            for key in extra_info_to_pack.keys():
                packed[key] = extra_info_to_pack[key]

        packed["file_path"] = file_path
        packed["file_num_tokens"] = file_original_tokens
        packed["is_truncated"] = is_truncated
        packed["response"] = responseJSON
        packed["model_str"] = model_str
        packed["total_cost"] = total_cost

        return packed


    # Generate a high-level summary of the repository using a language model, based on the README file.
    def generate_repo_summary_with_llm(self, file_path:str, model_str:str="gpt-4.1-mini"):
        MAX_TOKENS_PER_FILE = 50000

        # If the file was truncated, then `file_original_tokens` will be populated with the original number of tokens.
        def mkPrompt(file_contents:str, file_original_tokens:int=None, reflection:str=None):
            prompt = ""
            prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
            prompt += "\n"

            prompt += "# Task\n"
            prompt += "Your task is to generate a concise, information-rich summary (up to 10 sentences long) that precisely describes what this code repository is, what it's purpose is/what it's used for, and how it works.  This summary will be generated from the contents of the README file.\n"
            prompt += "In addition, if the documentation lists any specific files, runscripts, entry points, or code that is highly relevant to the task of setting up/running the code, and generating a code example, you should list those files in the `highly_relevant_files` key of your JSON response.\n"
            prompt += "\n"

            prompt += "# README file\n"
            if (file_original_tokens is not None):
                prompt += f"Note: Due to its size, this file was truncated to {MAX_TOKENS_PER_FILE} tokens, but it originally had {file_original_tokens} tokens.\n"
            prompt += "```\n"
            prompt += file_contents + "\n"
            prompt += "```\n"
            prompt += "\n"

            if (reflection is not None):
                prompt += "\n"
                prompt += "# Reflection\n"
                prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
                prompt += "```\n"
                prompt += reflection + "\n"
                prompt += "```\n"
                prompt += "\n"

            prompt += "# Why you are doing this\n"
            prompt += "You are part of an automated scientific discovery system that is working to rapidly accelerate the pace of scientific discovery, and generally make scientific discovery faster, less expensive, and more accessible.  Performing well at this task could have large positive benefits for humanity.  Performing poorly at this task is considered a critical failure, and will (at a minimum) cost time, money, and other resources, but could also lead to incorrect results (a strongly negative outcome).\n"
            prompt += "\n"

            prompt += "# Output Format\n"
            prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
            prompt += "\n"
            prompt += "Your JSON response must be a dictionary, that contains exactly the following key(s):\n"
            prompt += "- `description`(str): A brief description that is concise, information-rich summary (up to 10 sentences long) that precisely describes what this code repository is, what it's purpose is/what it's used for, and how it works.\n"
            prompt += "\n"
            prompt += "For example:\n"
            prompt += "```\n"
            prompt += "{\n"
            prompt += "  \"description\": \"concise, information-rich summary here\"\n"
            prompt += "  \"highly_relevant_files\": [\n"
            prompt += "    \"docs/setup.md\",\n"
            prompt += "    \"examples/example1.py\"\n"
            prompt += "  ],\n"
            prompt += "}\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "# Important Notes\n"
            prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
            prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
            prompt += "- All your information must be accurate. Do not hallucinate.\n"

            return prompt

        # Step 1: Read the file contents
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_contents = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

        # Step 2: Count the number of tokens in the file
        file_contents_tokens = countTokens(file_contents)
        file_original_tokens = file_contents_tokens
        # If the file is too large, then truncate it
        is_truncated = False
        if (file_contents_tokens > MAX_TOKENS_PER_FILE):
            file_contents = trimToMaxTokens(file_contents, MAX_TOKENS_PER_FILE)
            is_truncated = True

        # Step 3: Make the prompt
        prompt = ""
        if (is_truncated == False):
            prompt = mkPrompt(file_contents=file_contents, file_original_tokens=None)
        else:
            prompt = mkPrompt(file_contents=file_contents, file_original_tokens=file_original_tokens)

        # Check maximum token count
        MAX_TOKENS_FULL_PROMPT = 100000 - 5000
        if ("gpt-oss-120b" in model_str):
            MAX_TOKENS_FULL_PROMPT = 80000 - 5000
        prompt_tokens = countTokens(prompt)
        if (prompt_tokens > MAX_TOKENS_FULL_PROMPT):
            print(f"ERROR: Prompt is too long ({prompt_tokens} tokens, max {MAX_TOKENS_FULL_PROMPT}).  Skipping file {file_path}.")
            return None

        # Step 4: Call the language model
        total_cost = 0.0
        responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.0, jsonOut=False)
        total_cost += cost

        # Step 5: Parse the response
        if (responseJSON is None):
            print(f"Error: No response from the language model for file {file_path}")
            # Retry with a slightly higher temperature
            responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.1, jsonOut=False)
            total_cost += cost

        # Return
        packed = {
            "file_path": file_path,
            "response": responseJSON,
            "model": model_str,
            "total_cost": total_cost
        }

        return packed



    #
    #   Generate initial example from the code repository.
    #
    def generate_initial_example(self, repository_description:str, repo_url:str, selected_files:dict, model_str:str="o4-mini"):
        # Step 1: Generate a prompt to generate an initial example from the repository
        def mkPrompt(repository_description:str, repo_url:str, selected_files:list, file_contents:dict):
            prompt = ""
            prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
            prompt += "\n"

            prompt += "# Task\n"
            prompt += "Your task is to generate a complete, working code example that demonstrates the core functionality of a code repository.  The example should be concise, information-rich, and demonstrate how to use the code in the repository to perform its primary function.\n"
            prompt += "If the repository appears to have a variety of primary and auxiliary functions, you should focus on the primary function, but potentially also create separate example functions (example1(), example2(), example3(), etc.) that illustrate the other functionality.\n"
            prompt += "The purpose of these examples is to provide complete, functional, and informative examples to non-experts, so that the code just-works, and they can modify it to suit their purposes if needed.\n"
            prompt += "\n"
            prompt += "Previously, you were provided with each file in the repository (one at a time), and classified which files you believe are most relevant to this example generation task.\n"
            prompt += "Now, you will be provided with the most highly-rated files, and your task will be to generate a complete working code example that runs in a container.\n"
            prompt += "\n"

            prompt += "# Repository Description\n"
            prompt += "The repository URL is: " + repo_url + "\n"
            prompt += "\n"
            prompt += "A description of the repository (that was automatically generated from the README file) is as follows:\n"
            prompt += "```\n"
            prompt += repository_description + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# List of highly relevant files:\n"
            total_files = len(selected_files)
            prompt += "Here is a list of the " + str(total_files) + " most highly relevant files that you found in the repository (that will be provided below):\n"
            if (total_files == 0):
                prompt += "No files were classified as highly relevant.\n"
            else:
                file_idx = 1
                for file_info in selected_files:
                    prompt += "- File " + str(file_idx) + ": " + file_info["file_path_repo"]
                    prompt += "Metadata: " + json.dumps(file_info["response"]) + "\n"
                    prompt += "\n"
                    file_idx += 1

            prompt += "# File Contents\n"
            prompt += "Below is the contents of each of the highly relevant files.\n"
            prompt += "\n"
            for file_info in selected_files:
                # Get the filename (with path), to use as a key
                # Get the file contents
                try:
                    file_path = file_info.get("file_path_repo", None)
                    if (file_path is None):
                        continue
                    file_contents_str = file_contents.get(file_path, None)
                    if (file_contents_str is None):
                        print("ERROR: File contents not found for file: " + str(file_path))
                        continue
                    prompt += f"## File: {file_path}\n"
                    prompt += "Contents:\n"
                    prompt += "```\n"
                    prompt += file_contents_str + "\n"
                    prompt += "```\n"
                    prompt += "\n"
                except KeyError:
                    print("ERROR: File contents not found for file: " + str(file_path))
                    #file_contents_str = "File contents not found."

            prompt += "\n"

            prompt += "# Why you are doing this\n"
            prompt += "You are part of an automated scientific discovery system that is working to rapidly accelerate the pace of scientific discovery, and generally make scientific discovery faster, less expensive, and more accessible. Performing well at this task could have large positive benefits for humanity.  Performing poorly at this task is considered a critical failure, and will (at a minimum) cost time, money, and other resources, but could also lead to incorrect results (a strongly negative outcome).\n"
            prompt += "\n"

            prompt += "# Container Environment\n"
            prompt += "You will be generating a complete, working code example that runs in a pre-existing container.  Here is more information about the container:\n"
            prompt += "- The container is running Ubuntu 22.04.\n"
            prompt += "- You need to install the required Python version using conda (e.g. `conda create -y -n myenv python=3.12`). Conda is not installed by default, use the example to install miniconda.\n"
            prompt += "- You need to install the required Python dependencies using pip (e.g. `pip install -r requirements.txt`) after activating the conda environment.\n"
            prompt += "- Very minimal system-level packages, so you may need to install them using `apt-get install`.\n"
            prompt += "- Runtime: Currently, the maximum runtime of the container is " + str(self.max_runtime_mins) + " minutes, so you should frame your code to run within that time limit. If that is absolutely not possible for a technical reason, set a flag called `impossible_to_run_within_limit` flag to be `TRUE` in the metadata output.\n"
            prompt += "- Runtime (2): It's important to note that " + str(self.max_runtime_mins) + " minutes is an absolute maximum -- you are STRONGLY encouraged to have code examples that run as quickly as possible, while still being straightforward.\n"
            prompt += "\n"
            prompt += "## Additional container information\n"
            prompt += "- The container is CPU-only, and DOES NOT have any GPU resources.\n"
            prompt += "- The container does have internet access\n"
            prompt += "- The container does NOT have any API keys (such as OPENAI API keys), so it can not successfully make calls that require API keys.\n"
            prompt += "\n"


            prompt += "# Logging\n"
            prompt += "For demonstration and debugging purposes, your code must output the following output files in its working directory:\n"
            prompt += "- `log.json`: A JSON file that contains a log of what the code did at each time step, to help understand what the code is doing, and debug any issues that arise.\n"
            prompt += "- `results.json`: a JSON file that contains the final output/results of the code, to show that it worked successfully.\n"
            prompt += "- Anything you'd like to save to demontrate functionality, such as data, figures, etc., must be placed in a `to_save/` subdirectory off the working directory that `main.py` is in. The total size of these files should not be too large (e.g. ideally under 25MB total), or the download will timeout.\n"
            prompt += "Only the above files (`log.json`, `results.json`, and the contents of the `to_save/` directory) will be saved and returned to the user -- any other files created during runtime are ephemeral and will be deleted after execution, and not seen by the user.\n"
            prompt += "\n"

            prompt += "# Output format\n"
            prompt += "You are encouraged to think and plan before writing your code/output, but the code/output (contained within exactly 4 codeblocks) must be the last thing you write/output.\n"
            prompt += "You will need to generate exactly 4 output files, each enclosed in codeblock backticks (```).  Do NOT use backticks in your generated files as this will likely not parse correctly and be a critical failure.\n"
            prompt += "The 4 output files, in order, are:\n"
            prompt += "- A program (that will be saved as `main.py`) that contains the example code\n"
            prompt += "- A requirements file (saved as 'requirements.txt') that lists the dependencies, that will be automatically installed using pip.\n"
            prompt += "- A runscript (saved as `run.sh`) that will perform any environment-specific setup in the Ubuntu 22.04 container (e.g. cloning the original repository if required, installing any system-level packages (e.g. `apt-get install graphviz`), etc.)\n"
            #prompt += "- A set of metadata in JSON format (saved as `metadata.json`) that contains information about the example). It must be a dictionary with the following keys: `repository_url`:str, `description`:str, `example_description`:str, `example_type`:str, `user_type`:str, and `compute_requirements`:dict.  The `compute_requirements` key must be a dictionary with the following keys: `gpu`:bool, `min_gpu_memory_gb`:int, `cpu_cores`:int, `ram_gb`:int, `disk_space_gb`:int, `api_keys_required`:list[str], and `api_keys_optional`:list[str].  For unknown values, write none/null, not a blank string, or zero.\n"
            prompt += "- A set of metadata in JSON format (saved as `metadata.json`) that contains information about the example). It must be a dictionary with the following keys: `repository_url`:str, `description`:str, `example_description`:str, `example_inclusion_criteria`:str, `example_exclusion_criteria`:str, `example_type`:str, `user_type`:str, and `compute_requirements`:dict.  The `compute_requirements` key must be a dictionary with the following keys: `gpu_required`:bool, `gpu_normally_recommended`:bool, `min_gpu_memory_gb`:int, `cpu_cores`:int, `ram_gb`:int, `disk_space_gb`:int, `api_keys_required`:list[str], `api_keys_optional`:list[str], and `time_constraint_limited`:bool.  For unknown values, write none/null, not a blank string, or zero.\n"
            prompt += "\n"
            prompt += "## Example Output\n"
            prompt += "Below is a cartoon sketch of the type of example code that you should generate.\n"
            prompt += "First codeblock (main.py):\n"
            prompt += "```\n"
            prompt += "# Imports here\n"
            prompt += "\n"
            prompt + "# Describe what example 1 does here\n"
            prompt += "def example1():\n"
            prompt += "    # Code for example 1 here\n"
            prompt += "\n"
            prompt += "# Describe what example 2 does here\n"
            prompt += "def example2():\n"
            prompt += "    # Code for example 2 here\n"
            prompt += "\n"
            prompt += "# etc.\n"
            prompt += "\n"
            prompt += "# Main entry point (to run the examples)\n"
            prompt += "if __name__ == '__main__':\n"
            prompt += "    example1()\n"
            prompt += "    example2()\n"
            prompt += "    # etc.\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "Second codeblock (requirements.txt):\n"
            prompt += "```\n"
            prompt += "# List of dependencies here, one per line\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "Third codeblock (run.sh):\n"
            prompt += "```\n"
            prompt += "#!/bin/bash\n"
            prompt += "# Run any environment-specific setup here.  Note, your runscript must ALWAYS have the REQUIRED elements.\n"
            prompt += "# Example is below\n"
            prompt += "# Step 1: (OPTIONAL) If the original repository needs to be cloned, do that here\n"
            prompt += "git clone " + repo_url + "  # Example \n"
            prompt += "# Step 2: (OPTIONAL) Install any system-level packages here\n"
            prompt += "apt-get update && apt-get install -y graphviz # This is just an example -- don't install graphviz unless you need it!\n"
            # prompt += "# Step 3: (REQUIRED) Create the conda environment\n"
            # prompt += "conda create -y -n myenv python=3.12  # Select appropriate Python version\n"
            # prompt += "# Step 4: (REQUIRED) Activate the conda environment\n"
            # prompt += "source activate myenv\n"
            # prompt += "# Step 5: (REQUIRED) Install the Python dependencies\n"
            # prompt += "export PIP_ROOT_USER_ACTION=ignore\n"
            # prompt += "pip install -r requirements.txt\n"
            prompt += """# Step 3 (required): Minimal system packages for downloading Miniconda
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget bzip2 ca-certificates
rm -rf /var/lib/apt/lists/*

# Step 4 (required): Install Miniconda (if not already)
if ! command -v conda >/dev/null 2>&1; then
  echo "[RUN.SH] Installing Miniconda..."
  CONDA_DIR="/root/miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniconda.sh
  export PATH="$CONDA_DIR/bin:$PATH"
else
  echo "[RUN.SH] Conda already available."
  CONDA_DIR="$(dirname "$(dirname "$(command -v conda)")")"
  export PATH="$CONDA_DIR/bin:$PATH"
fi

# Initialize conda in this shell
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Step 5 (required): Configure conda to use conda-forge only (avoid Anaconda TOS issues)
echo "[RUN.SH] Configuring conda channels to use conda-forge only..."
conda config --system --set auto_update_conda false || true
conda config --system --set channel_priority strict || true
# Remove defaults/anaconda channels if present
conda config --system --remove channels defaults || true
conda config --system --remove-key default_channels || true
# Add conda-forge
conda config --system --add channels conda-forge || true

# Optional: attempt to accept TOS if the conda version supports it (best-effort)
if conda --help | grep -q "tos"; then
  echo "[RUN.SH] Attempting to accept Anaconda TOS (best-effort)..."
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

# Step 6 (required): Create the conda environment
ENV_NAME="myenv"
PY_VER="3.10"
if conda env list | grep -qE "^\\s*${ENV_NAME}\\s"; then
  echo "[RUN.SH] Conda env '${ENV_NAME}' already exists."
else
  echo "[RUN.SH] Creating conda env '${ENV_NAME}' with Python ${PY_VER} (from conda-forge)..."
  conda create -y -n "${ENV_NAME}" -c conda-forge python="${PY_VER}"
fi

# Step 7 (required): Activate env and install Python dependencies with pip
conda activate "${ENV_NAME}"
export PIP_ROOT_USER_ACTION=ignore

python -c "import sys; print('[RUN.SH] Python', sys.version)"
pip --version

echo "[RUN.SH] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

"""

            prompt += "# Step 8: Any other setup steps here\n"
            prompt += "# NOTE: Do *NOT* run the main program from here, that will be done separately.\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "Fourth codeblock (metadata.json):\n"
            prompt += "```\n"
            prompt += "{\n"
            prompt += "  \"repository_url\": \"" + repo_url + "\",\n"
            prompt += "  \"description\": \"" + repository_description + "\",\n"
            prompt += "  \"example_description\": \"This is a complete, working code example that...\", # A concise, information-dense, description of what what this code example demonstrates how to do, which would be useful both for humans to read but also search to lock onto important keywords or terms.\n"
            prompt += "  \"example_inclusion_criteria\": \"...\",  # A concise, information-dense description of the broad situations where this example would be useful to a user, perhaps solving particular kinds of tasks, and/or working with particular kinds of data, etc.\n"
            prompt += "  \"example_exclusion_criteria\": \"...\",  # A concise, information-dense description of the broad situations where this example would NOT be useful to a user, perhaps situations where it would fail, or not be applicable, etc.\n"
            prompt += "  \"example_type\": \"...\" # One of exactly the following 4 keys: `library_demo` (showing how to use libraries, like transformers), `application_demo` (showing how to use complete standalone programs, like dot/graphviz), `code_snippet_demo` (showing how to use complete self-contained code), or `other` (all other).\n"
            prompt += "  \"user_type\": \"...\". # One of exactly the following keys: `needs_user` (for code with user interfaces or that requires user input, or have interactive graphical interfaces), or `no_user` (for library examples, procedure demos, command-line tools that do not require user input, and can be successfully called/executed from Python code)\n"
            prompt += "  \"compute_requirements\": { # Use exactly these keys.  For unknown values, write none/null.\n"
            prompt += "    \"gpu_required\": true, # bool\n"
            prompt += "    \"gpu_normally_recommended\": true, # Set to `true` if the repository normally suggests/requires using a GPU, regardless of whether this code example requires one.\n"
            prompt += "    \"min_gpu_memory_gb\": 16, # int\n"
            prompt += "    \"cpu_cores\": 8, # int\n"
            prompt += "    \"ram_gb\": 32, # int \n"
            prompt += "    \"disk_space_gb\": 100, # int\n"
            prompt += "    \"api_keys_required\": [\"OPENAI_API_KEY\"], # list of str, of any API key *normally required* for primary functionality of the repository, regardless of whether required here.\n"
            prompt += "    \"api_keys_optional\": [\"SEMANTICSCHOLAR_API_KEY\"] # list of str, of any API key *optional* for primary functionality (i.e. Semantic Scholar keys are optional because it still allows access with no key, just at a slower rate limit),\n"
            prompt += "    \"time_constraint_limited\": false, # bool: Does this repository normally require more time to run than is available in the container (currently " + str(self.max_runtime_mins) + " minutes), and due to this, a limited version was run? Would having more time enable significantly better/more accurate results? If so, `true`, otherwise, `false`.\n"
            prompt += "  }\n"
            prompt += "}\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "# Important Notes\n"
            prompt += "- You must return exactly 4 codeblocks, in the order specified above.\n"
            prompt += "- You are encouraged to think and plan before writing your code/output, but the code/output (contained within exactly 4 codeblocks) must be the last thing you write/output, or it will not be parsed correctly.\n"
            prompt += "- You must not use backticks in your generated files, as this will likely not parse correctly and be a critical failure.\n"
            prompt += "- The code must be complete, functional, work on the first try in a Ubuntu 22.04 container, informative, and demonstrate how to use the code in the repository to perform its primary function.\n"
            prompt += "- The examples must NOT be trivial, they must be useful, clear demonstrations of utility.\n"
            prompt += "- The examples should ideally illustrate the main function of the repository -- the things a normal user is likely to do with it -- and not focus on edge cases or unusual functionality.\n"
            prompt += "- The top-level function names of the examples must be `example1()`, `example2()`, `example3()`, etc.  The main entry point of the code must be `if __name__ == '__main__':`.\n"
            prompt += "- The examples are intended to be USEFUL and MODIFIABLE by modifying the code example itself. If you just output a function that calls an existing demo or example, that is not useful or acceptable, as it will not be easily modifiable. You're trying to show a non-expert how to use this code, in a way that they can easily modify to use for themselves.\n"
            prompt += "- If the code could reasonably be expected to require API keys, list them int he `api_keys_required` or `api_keys_optional` keys in the metadata, as appropriate.  This is very important, otherwise we won't know what code we can run in the container, and will waste time and money trying to run code that can't run.\n"
            prompt += "- There should be very detailed (but concise, information-dense) documentation at the top of the main.py file, describing exactly what input the code expects (with examples), what output it produces (with examples), and any other relevant information to getting it to run for a non-expert.\n"
            prompt += "- Your code should VERBOSELY output what it is doing, so that a non-expert can understand what is happening, but also debug it if something is not working as expected.  It should also provide an EASY, OBVIOUS mechanism to verify that the code is successfully accomplishing it's goal correctly and faithfully.\n"
            prompt += "- All your information must be accurate. Do not hallucinate.\n"

            return prompt

        # Step 1: Pick out the most highly relevant files from the sorted file classifications, up to the maximum number of tokens.
        MAX_TOKENS = 100000     # 100k tokens total (across all files)
        if ("gpt-oss-120b" in model_str):
            MAX_TOKENS = 60000   # 60k tokens total (across all files)

        # Flatten the list of files
        all_files = selected_files
        # Remove any files that don't have a relevance rating (and that are not none)
        #all_files = [file for file in all_files if "how_relevant_rating" in file["response"]]
        all_files = [file for file in all_files if file.get("response", {}).get("how_relevant_rating", None) is not None]
        # Sort the files by how relevant rating, descending
        all_files.sort(key=lambda x: x["response"]["how_relevant_rating"], reverse=True)
        # Pick out the most highly relevant files, up to the maximum number of tokens
        selected_files = []
        total_tokens = 0
        file_content = {}
        TRUNCATED_SIZE = 10000  # If we truncate a file, truncate it to 10k tokens (to give room for other files in the context)
        for file in all_files:
            #print("Parsing file:")
            #print(json.dumps(file, indent=4, ensure_ascii=False))
            # Use encoder for Enum
            #print(json.dumps(file, indent=4, default=lambda o: o.value if isinstance(o, enum.Enum) else o))
            #print("\n")

            # First, try to load the file, to place it in the file content dictionary
            file_contents = None
            file_rating = file["response"]["how_relevant_rating"]
            # If the rating is less than 3, skip it
            if (file_rating < 3):
                #print(f"Skipping file {file['file_path_repo']} due to low relevance rating ({file_rating})")
                continue

            try:
                file_path_local = file["file_path_local"]
                file_path_repo = file["file_path_repo"]
                with open(file_path_local, 'r', encoding='utf-8') as f:
                    file_contents = f.read()
                file_content[file_path_repo] = file_contents
            except Exception as e:
                print(f"Error reading file {file_path_local}: {e}")

            if (file_contents is None):
                print(f"Error: File contents not found for file {file_path_local}")
                continue    # Do not continue with this file, since we can't use it

            # Add the file contents to the file content dictionary
            file_tokens = file["file_num_tokens"]
            # Check how many tokens are remaining
            remaining_tokens = MAX_TOKENS - total_tokens
            if (file_tokens <= remaining_tokens):
                # Add the file to the selected files
                selected_files.append(file)
                total_tokens += file_tokens
            # Check if there's enough space for a truncated version
            elif (remaining_tokens >= TRUNCATED_SIZE):
                # Truncate the file contents to the remaining tokens
                file_contents = trimToMaxTokens(file_contents, remaining_tokens)
                # Add the file to the selected files, with the truncated contents
                file_content[file_path_repo] = file_contents + "\n### NOTE: THIS LONG FILE WAS TRUNCATED TO FIT WITHIN THE AVAILABLE SPACE ###\n"
                selected_files.append(file)
                total_tokens += TRUNCATED_SIZE
            else:
                # Not enough space for this file, skip it
                print(f"Skipping file {file['file_path_repo']} due to insufficient space (remaining tokens: {remaining_tokens}, file tokens: {file_tokens})")
                continue


        # Step 2: Generate the prompt to generate the initial example
        prompt = mkPrompt(repository_description=repository_description, repo_url=repo_url, selected_files=selected_files, file_contents=file_content)

        # Check maximum token count
        MAX_TOKENS_FULL_PROMPT = 150000 - 5000
        if ("gpt-oss-120b" in model_str):
            MAX_TOKENS_FULL_PROMPT = 80000 - 5000
        prompt_tokens = countTokens(prompt)
        if (prompt_tokens > MAX_TOKENS_FULL_PROMPT):
            print(f"ERROR: Prompt is too long ({prompt_tokens} tokens, max {MAX_TOKENS_FULL_PROMPT}).")
            return None

        # Step 3: Call the language model to generate the initial example
        total_cost = 0.0
        #model_str_code_generation = "o3-mini"
        #model_str_code_generation = "o4-mini"
        responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.0, jsonOut=False)
        total_cost += cost

        # Step 4: Parse the response (the text) into 4 codeblocks
        if (responseText is None):
            print("Error: No response from the language model for generating the initial example")
            return None
        codeblocks = find_codeblocks(responseText)
        print("Returned " + str(len(codeblocks)) + " codeblocks")

        if (len(codeblocks) != 4):
            print("Error: Expected 4 codeblocks, but got " + str(len(codeblocks)) + ".  This is a critical failure.")

            # Retry with a slightly higher temperature
            prompt += "\n\nIMPORTANT NOTE: The previous response did not follow the instructions correctly, and did not return exactly 4 codeblocks.  Please try again, and ensure that you return exactly 4 codeblocks, in the order specified, and do not use backticks in your generated files, as this will likely not parse correctly and be a critical failure.\n"
            responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.1, jsonOut=False)
            total_cost += cost
            if (responseText is None):
                print("Error: No response from the language model for generating the initial example (2nd attempt)")
                return None
            codeblocks = find_codeblocks(responseText)
            print("Returned " + str(len(codeblocks)) + " codeblocks (2nd attempt)")

            if (len(codeblocks) != 4):
                print("Error: Expected 4 codeblocks, but got " + str(len(codeblocks)) + " (2nd attempt).  This is a critical failure.")
                return None

        code = codeblocks[0]
        requirements = codeblocks[1]
        runscript = codeblocks[2]
        metadata = codeblocks[3]
        # Try to parse the metadata JSON
        metadata_json = None
        # try:
        #     metadata_json = json.loads(metadata)
        # except json.JSONDecodeError as e:
        #     print(f"Error parsing metadata JSON: {e}")
        #     return None
        metadata_json, cost_fix = parse_and_fix_json(metadata, model_str="gpt-5-mini")
        total_cost += cost_fix

        # Store the total cost of this generation
        self.costs["generate_initial_code_example"] = total_cost

        # Return
        packed = {
            "repo_url": repo_url,
            "repository_description": repository_description,
            "model_str": model_str,
            "total_cost": total_cost,
            "code": code,
            "requirements": requirements,
            "runscript": runscript,
            "metadata": metadata_json,
        }
        return packed


    #
    #   Running code examples in a container
    #
    def run_code_example_in_container(self, code:str, requirements:str, runscript:str, metadata:dict, path_out:str, timeout_sec:int=60*15):
        from modules.ModuleRunPythonInModal import ModuleRunPythonInModal
        moduleModal = ModuleRunPythonInModal()

        # Create the payload
        payload = {
            "input": {
                "base_path": path_out,
                "max_runtime_seconds": timeout_sec,
                "python_version": "3.12",
                "requirements.txt": requirements,
                "code": code,
                "runscript": runscript,
            }
        }

        # Run the action
        result = moduleModal.runAction(moduleModal.ACTION_RUN_PROGRAM["name"], payload)
        print("run_code_in_container result:")
        print(json.dumps(result, indent=4))

        # Pack current code/output
        packed = {
            "path_out": path_out,
            "code": code,
            "requirements": requirements,
            "runscript": runscript,
            "metadata": metadata,
            "result": result,
            "pip.stdout": result.get("output", {}).get("pip.stdout", None),
            "pip.stderr": result.get("output", {}).get("pip.stderr", None),
            "python.stdout": result.get("output", {}).get("python.stdout", None),
            "python.stderr": result.get("output", {}).get("python.stderr", None),
            "runscript.stdout": result.get("output", {}).get("runscript.stdout", None),
            "runscript.stderr": result.get("output", {}).get("runscript.stderr", None),
            "log": result.get("output", {}).get("log", None),
            "results_json": result.get("output", {}).get("results_json", None),
            "other_errors": result.get("other_errors", None),
        }

        return packed


    #
    #   Reflecting/Debugging a code example
    #
    #
    #   Generate initial example from the code repository.
    #
    #def reflect_code_example(self, repository_description:str, repo_url:str, selected_files:dict, current_example_state:dict):
    def reflect_code_example(self, current_example_state:dict, model_str:str="o4-mini"):
        # Step 1: Generate a prompt to generate an initial example from the repository
        def mkPromptErrorDetection(repository_description:str, repo_url:str, selected_files:list, file_contents:dict, current_example_state:dict, MAX_TOKENS_FULL_PROMPT:int=145000):
            prompt = ""
            prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
            prompt += "\n"

            prompt += "# Task\n"
            prompt += "Your task is to generate a complete, working code example that demonstrates the core functionality of a code repository.  The example should be concise, information-rich, and demonstrate how to use the code in the repository to perform its primary function.\n"
            prompt += "If the repository appears to have a variety of primary and auxiliary functions, you should focus on the primary function, but potentially also create separate example functions (example1(), example2(), example3(), etc.) that illustrate the other functionality.\n"
            prompt += "The purpose of these examples is to provide complete, functional, and informative examples to non-experts, so that the code just-works, and they can modify it to suit their purposes if needed.\n"
            prompt += "\n"
            prompt += "Previously, you were provided with highly relevant files in the repository, and asked to generate a code example.\n"
            prompt += "That code example was executed, and the results are below.\n"
            prompt += "Now, your task is to reflect on the code example and it's output/results, and determine if it ran successfully, and if not, determine what went wrong and fix any errors.\n"
            prompt += "To assist with this task, the following will be provided below:\n"
            prompt += "- The repository description, which was automatically generated from the README file.\n"
            prompt += "- The highly relevant files previously identified in the repository\n"
            prompt += "- The example code you generated (including the code, requirements, and runscript)\n"
            prompt += "- The output/results of the code example, including any errors or issues that were encountered.\n"
            prompt += "\n"

            prompt += "# Repository Description\n"
            prompt += "The repository URL is: " + repo_url + "\n"
            prompt += "\n"
            prompt += "A description of the repository (that was automatically generated from the README file) is as follows:\n"
            prompt += "```\n"
            prompt += repository_description + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# List of highly relevant files from the repository:\n"
            total_files = len(selected_files)
            prompt += "Here is a list of the " + str(total_files) + " most highly relevant files that you found in the repository (that will be provided below):\n"
            if (total_files == 0):
                prompt += "No files were classified as highly relevant.\n"
            else:
                file_idx = 1
                for file_info in selected_files:
                    # print("File info (file idx): " + str(file_idx))
                    # print(json.dumps(file_info, indent=4, ensure_ascii=False, default=lambda o: o.value if isinstance(o, enum.Enum) else o))
                    # print("\n")

                    prompt += "- File " + str(file_idx) + ": " + file_info["file_path_repo"]
                    prompt += "Metadata: " + json.dumps(file_info["response"]) + "\n"
                    prompt += "\n"
                    file_idx += 1

            prompt += "# File Contents\n"
            prompt += "Below is the contents of each of the highly relevant files.\n"
            prompt += "\n"
            for file_info in selected_files:
                # Get the filename (with path), to use as a key
                # Get the file contents
                try:
                    file_path = file_info.get("file_path_repo", None)
                    if (file_path is None):
                        continue
                    file_contents_str = file_contents.get(file_path, None)
                    if (file_contents_str is None):
                        print("ERROR: File contents not found for file: " + str(file_path))
                        continue
                    prompt += f"## File: {file_path}\n"
                    prompt += "Contents:\n"
                    prompt += "```\n"
                    prompt += file_contents_str + "\n"
                    prompt += "```\n"
                    prompt += "\n"
                except KeyError:
                    print("ERROR: File contents not found for file: " + str(file_path))
                    #file_contents_str = "File contents not found."

            prompt += "\n"

            prompt += "# Your previously generated code example\n"
            prompt += "Below is the example code that you generated, including the code, requirements, and runscript.\n"
            prompt += "\n"
            prompt += "## Code (main.py)\n"
            prompt += "```\n"
            prompt += current_example_state.get("code", "ERROR: No code found. This is a critical error!") + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## requirements.txt\n"
            prompt += "```\n"
            prompt += current_example_state.get("requirements", "ERROR: No requirements found. This is a critical error!") + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## run.sh\n"
            prompt += "```\n"
            prompt += current_example_state.get("runscript", "ERROR: No runscript found. This is a critical error!") + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# Change log\n"
            prompt += "Below is the change log (between debugging iterations) of the example code.\n"
            prompt += "The maximum number of debugging iterations is: " + str(self.max_debug_iterations) + "\n"
            prompt += "This can help you determine what issues you previously identified/fixes you tried, to avoid repeating actions that are ineffectual.\n"
            change_log = current_example_state.get("change_log", [])
            prompt += "```\n"
            if (len(change_log) == 0):
                prompt += "No changelog found. This is likely the first debugging iteration.\n"
            else:
                prompt += json.dumps(change_log, indent=4, ensure_ascii=False) + "\n"
            prompt += "```\n"
            prompt += "\n"


            prompt += "# Output/results of the code example\n"
            prompt += "Below is the output/results of the code example you generated, including any errors or issues that were encountered.\n"
            prompt += "You will need to analyze this output/results to determine if the code example ran successfully, and if not, determine what went wrong and fix any errors.\n"
            prompt += "\n"

            # Token limit per log file
            MAX_TOKENS_PER_LOG_FILE = 10000
            # Adaptively adjust the max tokens per log file based on the total number of tokens in the prompt
            #MAX_TOKENS_FULL_PROMPT = 150000 - 5000  # 150k tokens total, minus a buffer for the rest of the prompt
            #if ("gpt-oss-120b" in model_str):
            #    MAX_TOKENS_FULL_PROMPT = 75000 - 5000       # ~130k total tokens, but that includes the response as well.

            total_prompt_tokens = countTokens(prompt)
            tokens_left = MAX_TOKENS_FULL_PROMPT - total_prompt_tokens
            MAX_TOKENS_PER_LOG_FILE = min(MAX_TOKENS_PER_LOG_FILE, tokens_left // 8)  # Divide by 5 to leave room for other logs
            if (MAX_TOKENS_PER_LOG_FILE > 10000):
                # Clip back down to 10k tokens if the limit exceeds that.
                MAX_TOKENS_PER_LOG_FILE = 10000
            if (MAX_TOKENS_PER_LOG_FILE < 1000):
                # If the limit is too low, set it to 1k tokens
                print("WARNING: MAX_TOKENS_PER_LOG_FILE is very low (" + str(MAX_TOKENS_PER_LOG_FILE) + "), this may lead to highly truncated logs.")
                MAX_TOKENS_PER_LOG_FILE = 1000

            execution_results = current_example_state.get("execution_result", [])
            last_execution_result = execution_results[-1] if execution_results else {}
            # Python stdout/stderr
            prompt += "## Python stdout\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("python.stdout", "ERROR: No Python stdout found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("python.stdout", "ERROR: No Python stdout found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            python_stdout = last_execution_result.get("python.stdout", "ERROR: No Python stdout found.\n")
            if (python_stdout is None):
                python_stdout = "ERROR: No Python stdout found.\n"
            prompt += trimPromptComponentLog(python_stdout.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## Python stderr\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("python.stderr", "ERROR: No Python stderr found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("python.stderr", "ERROR: No Python stderr found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            python_stderr = last_execution_result.get("python.stderr", "ERROR: No Python stderr found.\n")
            if (python_stderr is None):
                python_stderr = "ERROR: No Python stderr found.\n"
            prompt += trimPromptComponentLog(python_stderr.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # pip stdout/stderr
            prompt += "## PIP stdout\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("pip.stdout", "ERROR: No PIP stdout found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("pip.stdout", "ERROR: No PIP stdout found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            pip_stdout = last_execution_result.get("pip.stdout", "ERROR: No PIP stdout found.\n")
            if (pip_stdout is None):
                pip_stdout = "ERROR: No PIP stdout found.\n"
            prompt += trimPromptComponentLog(pip_stdout.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## PIP stderr\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("pip.stderr", "ERROR: No PIP stderr found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("pip.stderr", "ERROR: No PIP stderr found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            pip_stderr = last_execution_result.get("pip.stderr", "ERROR: No PIP stderr found.\n")
            if (pip_stderr is None):
                pip_stderr = "ERROR: No PIP stderr found.\n"
            prompt += trimPromptComponentLog(pip_stderr.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Runscript stdout/stderr
            prompt += "## Run script stdout\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("runscript.stdout", "ERROR: No runscript stdout found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("runscript.stdout", "ERROR: No runscript stdout found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            runscript_stdout = last_execution_result.get("runscript.stdout", "ERROR: No runscript stdout found.\n")
            if (runscript_stdout is None):
                runscript_stdout = "ERROR: No runscript stdout found.\n"
            prompt += trimPromptComponentLog(runscript_stdout.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## Run script stderr\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("runscript.stderr", "ERROR: No runscript stderr found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("runscript.stderr", "ERROR: No runscript stderr found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            runscript_stderr = last_execution_result.get("runscript.stderr", "ERROR: No runscript stderr found.\n")
            if (runscript_stderr is None):
                runscript_stderr = "ERROR: No runscript stderr found.\n"
            prompt += trimPromptComponentLog(runscript_stderr.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Log file
            prompt += "## Log file (log.json)\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("log", "ERROR: No log file found. This is a critical error.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("log", "ERROR: No log file found. This is a critical error.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            log_file = last_execution_result.get("log", "ERROR: No log file found. This is a critical error.\n")
            if (log_file is None):
                log_file = "ERROR: No log file found. This is a critical error.\n"
            # Check the type
            if (isinstance(log_file, list) != True):
                # If it's not a list, convert it into something that can be split
                if (isinstance(log_file, dict)):
                    # If it's a dictionary, convert it to a string
                    log_file = json.dumps(log_file, indent=4, ensure_ascii=False)
                else:
                    log_file = str(log_file)
                log_file = log_file = log_file.split("\n")
            prompt += trimPromptComponentLog(log_file, maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Results file
            prompt += "## Results file (results.json)\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("results", "ERROR: No results file found. This is a critical error.\n"))
            #prompt += last_execution_result.get("results", "ERROR: No results file found. This is a critical error.\n")
            results_file = last_execution_result.get("results_json", "ERROR: No results file found. This is a critical error.\n")
            if (results_file is None):
                results_file = "ERROR: No results file found. This is a critical error.\n"
            # Check the type
            if (isinstance(results_file, list) != True):
                # If it's not a list, convert it into something that can be split
                if (isinstance(results_file, dict)):
                    # If it's a dictionary, convert it to a string
                    results_file = json.dumps(results_file, indent=4, ensure_ascii=False)
                else:
                    results_file = str(results_file)
                results_file = results_file.split("\n")
            prompt += trimPromptComponentLog(results_file, maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Container-specific errors
            prompt += "## Other container-specific errors (if any)\n"
            prompt += "```\n"
            prompt += str(last_execution_result.get("other_errors", "ERROR: No other errors found.\n"))
            prompt += "```\n"
            prompt += "\n"


            prompt += "# Why you are doing this\n"
            prompt += "You are part of an automated scientific discovery system that is working to rapidly accelerate the pace of scientific discovery, and generally make scientific discovery faster, less expensive, and more accessible. Performing well at this task could have large positive benefits for humanity.  Performing poorly at this task is considered a critical failure, and will (at a minimum) cost time, money, and other resources, but could also lead to incorrect results (a strongly negative outcome).\n"
            prompt += "\n"

            prompt += "# Container Environment\n"
            prompt += "You will be generating a complete, working code example that runs in a pre-existing container.  Here is more information about the container:\n"
            prompt += "- The container is running Ubuntu 22.04.\n"
            prompt += "- You need to install the required Python version using conda (e.g. `conda create -y -n myenv python=3.12`). Conda is not installed by default, use the example to install miniconda.\n"
            prompt += "- You need to install the required Python dependencies using pip (e.g. `pip install -r requirements.txt`) after activating the conda environment.\n"
            prompt += "- Very minimal system-level packages, so you may need to install them using `apt-get install`.\n"
            prompt += "- Runtime: Currently, the maximum runtime of the container is " + str(self.max_runtime_mins) + " minutes, so you should frame your code to run within that time limit. If that is absolutely not possible for a technical reason, set a flag called `impossible_to_run_within_limit` flag to be `TRUE` in the metadata output.\n"
            prompt += "- Runtime (2): It's important to note that " + str(self.max_runtime_mins) + " minutes is an absolute maximum -- you are STRONGLY encouraged to have code examples that run as quickly as possible, while still being straightforward.\n"
            prompt += "\n"
            prompt += "## Additional container information\n"
            prompt += "- The container is CPU-only, and DOES NOT have any GPU resources.\n"
            prompt += "- The container does have internet access\n"
            prompt += "- The container does NOT have any API keys (such as OPENAI API keys), so it can not successfully make calls that require API keys.\n"
            prompt += "\n"

            prompt += "# Logging\n"
            # prompt += "For demonstration and debugging purposes, your code must output the following output files in its working directory:\n"
            # prompt += "- `log.json`: A JSON file that contains a log of what the code did at each time step, to help understand what the code is doing, and debug any issues that arise.\n"
            # prompt += "- `results.json`: a JSON file that contains the final output/results of the code, to show that it worked successfully.\n"
            # prompt += "\n"
            prompt += "For demonstration and debugging purposes, your code must output the following output files in its working directory:\n"
            prompt += "- `log.json`: A JSON file that contains a log of what the code did at each time step, to help understand what the code is doing, and debug any issues that arise.\n"
            prompt += "- `results.json`: a JSON file that contains the final output/results of the code, to show that it worked successfully.\n"
            prompt += "- Anything you'd like to save to demontrate functionality, such as data, figures, etc., must be placed in a `to_save/` subdirectory off the working directory that `main.py` is in. The total size of these files should not be too large (e.g. ideally under 25MB total), or the download will timeout.\n"
            prompt += "Only the above files (`log.json`, `results.json`, and the contents of the `to_save/` directory) will be saved and returned to the user -- any other files created during runtime are ephemeral and will be deleted after execution, and not seen by the user.\n"
            prompt += "\n"


            prompt += "# Output format\n"
            prompt += "Your sole task in this step is to examine the example code and all it's output, and determine if it is running CORRECTLY, FAITHFULLY, ACCURATELY, and WITHOUT ERROR.\n"
            prompt += "If the code is meeting these requirements, then we will stop debugging the code, and give it to the end user.\n"
            prompt += "If the code is not meeting these requirements, then we will identify the issues, and try to modify the code to fix any remaining issues.\n"
            prompt += "Here is a non-exhaustive list of example issues:\n"
            prompt += "- The code is exiting with an error\n"
            prompt += "- There are errors in the log files, stdout/stderr, or results.\n"
            prompt += "- The logs or results to verify that the output is working as expected are missing.\n"
            prompt += "- The code appears to be running, but the output doesn't make sense, or isn't as it should be, suggesting logical errors or that the computation is incorrect.\n"
            prompt += "- The code is not running completely within the allocated time.\n"
            prompt += "- and so forth (many other errors are possible)\n"
            prompt += "\n"
            prompt += "Your output must be a JSON dictionary that contains the following keys:\n"
            prompt += "- `correct_behavior`:list(str): A list of strings, where each string (concisely) describes the successful behaviors/functionality the code is exhibiting.\n"
            prompt += "- `critical_issues`:list(str): A list of strings, where each string (concisely but very specifically, enough to pinpoint the issue for the debugging agent) describes one *critical issue* that has been identified that needs to be fixed.  A critical issue effects the functionality, correctness, or other critical factor of the example. The list should be exhaustive -- if you identify 5 issues, write all 5, not just 2 or 3.  Critical issues are those that prevent the code from running successfully, or that cause it to produce incorrect results.\n"
            prompt += "- `issues`:list(str): A list of strings, where each string (concisely but very specifically, enough to pinpoint the issue for the debugging agent) describes one issue that has been identified that needs to be fixed.  The list should be exhaustive -- if you identify 5 issues, write all 5, not just 2 or 3.\n"
            prompt += "- `is_faithful`:bool: A boolean identifying whether the code is executing faithfully to the task, not a watered-down version with fake data, stubs, or other issues.\n"
            prompt += "- `is_accurate`:bool: A boolean identifying whether the code is executing accurately and without error.\n"
            prompt += "- `is_complete`:bool: A boolean identifying whether the code is complete, functioning correctly, and clearly demonstrates the main functionality of the code/library/etc. in a way that would be useful for a non-expert to modify to use for their own purposes.\n"
            prompt += "- `faithfulness_accurate_completeness_statement`:str: A short statement that summarizes any issues with faithfulness, accuracy, or completeness of the code, that are off from the ideal.  This should be a concise statement that summarizes the issues, and should be written in a way that is easy to understand for a non-expert.\n"
            prompt += "- `is_finished`:bool: A boolean identifying whether the code is complete, functioning correctly, executing faithfully to the task, executing accurately and without error, and clearly demonstrates the main functionality of the code/library/etc. in a way that would be useful for a non-expert to modify to use for their own purposes. If there are any remaining issues, this should be marked as `false`. If it is marked as `true`, then the debugging process will end, and the code will be given to the user.  Having remaining errors/issues in the code will be a critical error, and constitute a large waste of time, money, and other resources.\n"
            prompt += "- `metadata`:dict: A set of metadata that contains information about the example. It must be a dictionary with the following keys: `repository_url`:str, `description`:str, `example_description`:str, `example_inclusion_criteria`:str, `example_exclusion_criteria`:str, `example_type`:str, `user_type`:str, and `compute_requirements`:dict. The `compute_requirements` key must be a dictionary with the following keys: `gpu`:bool, `min_gpu_memory_gb`:int, `cpu_cores`:int, `ram_gb`:int, and `disk_space_gb`:int.  For unknown values, write none/null, not a blank string, or zero.\n"
            prompt += "\n"
            prompt += "You can think as much as you like before producing the JSON response, but the JSON response above must be a valid JSON dictionary contained between codeblocks (```), and it must be the last thing you output. (It will be parsed automatically, and if it does not adhere to these format expectations then that will be a critical failure).\n"
            prompt += "\n"
            prompt += "## Example Output\n"
            prompt += "Below is a cartoon sketch of the output:\n"
            prompt += "```\n"
            prompt += "{\n"
            prompt += " \"correct_behavior\": [], # List of strings\n"
            prompt += " \"critical_issues\": [], # List of strings\n"
            prompt += " \"issues\": [], # List of strings\n"
            prompt += " \"is_faithful\": false # Boolean, whether the code is executing faithfully to the task, not a watered-down version with fake data, stubs, or other issues.\n"
            prompt += " \"is_accurate\": false # Boolean, whether the code is executing accurately and without error\n"
            prompt += " \"is_complete\": false # Boolean, whether the code is complete, functioning correctly, and clearly demonstrates the main functionality of the code/library/etc. in a way that would be useful for a non-expert to modify to use for their own purposes.\n"
            prompt += " \"faithfulness_accurate_completeness_statement\": \"...\" # A short statement that summarizes any issues with faithfulness, accuracy, or completeness of the code, that are off from the ideal.\n"
            prompt += " \"is_finished\": false # Boolean\n"
            prompt += " \"metadata\": {\n"
            prompt += "  \"repository_url\": \"" + repo_url + "\",\n"
            prompt += "  \"description\": \"" + repository_description + "\",\n"
            prompt += "  \"example_description\": \"This is a complete, working code example that...\", # A concise, information-dense, description of what what this code example demonstrates how to do, which would be useful both for humans to read but also search to lock onto important keywords or terms.\n"
            prompt += "  \"example_inclusion_criteria\": \"...\",  # A concise, information-dense description of the broad situations where this example would be useful to a user, perhaps solving particular kinds of tasks, and/or working with particular kinds of data, etc.\n"
            prompt += "  \"example_exclusion_criteria\": \"...\",  # A concise, information-dense description of the broad situations where this example would NOT be useful to a user, perhaps situations where it would fail, or not be applicable, etc.\n"
            prompt += "  \"example_type:\": \"...\" # One of exactly the following 4 keys: `library_demo` (showing how to use libraries, like transformers), `application_demo` (showing how to use complete standalone programs, like dot/graphviz), `code_snippet_demo` (showing how to use complete self-contained code), or `other` (all other).\n"
            prompt += "  \"user_type\": \"...\". # One of exactly the following keys: `needs_user` (for code with user interfaces or that requires user input, or have interactive graphical interfaces), or `no_user` (for library examples, procedure demos, command-line tools that do not require user input, and can be successfully called/executed from Python code)\n"
            prompt += "  \"compute_requirements\": { # Use exactly these keys.  For unknown values, write none/null.\n"
            prompt += "    \"gpu_required\": true, # bool\n"
            prompt += "    \"gpu_normally_recommended\": true, # Set to `true` if the repository normally suggests/requires using a GPU, regardless of whether this code example requires one.\n"
            prompt += "    \"min_gpu_memory_gb\": 16, # int\n"
            prompt += "    \"cpu_cores\": 8, # int\n"
            prompt += "    \"ram_gb\": 32, # int \n"
            prompt += "    \"disk_space_gb\": 100, # int\n"
            prompt += "    \"api_keys_required\": [\"OPENAI_API_KEY\"], # list of str, of any API key *normally required* for primary functionality of the repository, regardless of whether required here.\n"
            prompt += "    \"api_keys_optional\": [\"SEMANTICSCHOLAR_API_KEY\"] # list of str, of any API key *optional* for primary functionality (i.e. Semantic Scholar keys are optional because it still allows access with no key, just at a slower rate limit),\n"
            prompt += "    \"time_constraint_limited\": false, # bool: Does this repository normally require more time to run than is available in the container (currently " + str(self.max_runtime_mins) + " minutes), and due to this, a limited version was run? Would having more time enable significantly better/more accurate results? If so, `true`, otherwise, `false`.\n"
            prompt += "  }\n"
            prompt += "}\n"


            prompt += "}\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "# Important Notes\n"
            prompt += "- You must return exactly 1 codeblock\n"
            prompt += "- You are encouraged to think and plan before writing your code/output, but the JSON output (contained within the codeblock) must be the last thing you write/output, or it will not be parsed correctly.\n"
            prompt += "- The code must be complete, functional, work on the first try in a Ubuntu 22.04 container, informative, and demonstrate how to use the code in the repository to perform its primary function.\n"
            prompt += "- The examples must NOT be trivial, they must be useful, clear demonstrations of utility.\n"
            prompt += "- The examples should ideally illustrate the main function of the repository -- the things a normal user is likely to do with it -- and not focus on edge cases or unusual functionality.\n"
            prompt += "- The top-level function names of the examples must be `example1()`, `example2()`, `example3()`, etc.  The main entry point of the code must be `if __name__ == '__main__':`.\n"
            prompt += "- The examples are intended to be USEFUL and MODIFIABLE by modifying the code example itself. If you just output a function that calls an existing demo or example, that is not useful or acceptable, as it will not be easily modifiable. You're trying to show a non-expert how to use this code, in a way that they can easily modify to use for themselves.\n"
            prompt += "- Please remember the overall goal of this task is to generate complete, working examples.  If, when debugging, a complex example doesn't appear to be working, you should consider backing-off to a simpler example that's still useful.\n"
            prompt += "- If the code could reasonably be expected to require API keys, list them int he `api_keys_required` or `api_keys_optional` keys in the metadata, as appropriate.  This is very important, otherwise we won't know what code we can run in the container, and will waste time and money trying to run code that can't run.\n"
            prompt += "- Your code should VERBOSELY output what it is doing, so that a non-expert can understand what is happening, but also debug it if something is not working as expected.  It should also provide an EASY, OBVIOUS mechanism to verify that the code is successfully accomplishing it's goal correctly and faithfully.\n"
            prompt += "- There should be very detailed (but concise, information-dense) documentation at the top of the main.py file, describing exactly what input the code expects (with examples), what output it produces (with examples), and any other relevant information to getting it to run for a non-expert.\n"
            prompt += "- All the codeblocks must be properly terminated (``` at the start and end).  Any JSON must be valid and properly formatted JSON.  Your output will be automatically parsed, and any deviations in the output format will cause a crash, which is a critical error.\n"
            prompt += "- The final line in your output should be a triple backtick (```), indicating the end of the codeblock (the JSON).\n"
            prompt += "- All your information must be accurate. Do not hallucinate.\n"

            return prompt



        def mkPromptCodeReflection(repository_description:str, repo_url:str, selected_files:list, file_contents:dict, current_example_state:dict, error_information:list, MAX_TOKENS_FULL_PROMPT:int=145000):
            prompt = ""
            prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
            prompt += "\n"

            prompt += "# Task\n"
            prompt += "Your task is to generate a complete, working code example that demonstrates the core functionality of a code repository.  The example should be concise, information-rich, and demonstrate how to use the code in the repository to perform its primary function.\n"
            prompt += "If the repository appears to have a variety of primary and auxiliary functions, you should focus on the primary function, but potentially also create separate example functions (example1(), example2(), example3(), etc.) that illustrate the other functionality.\n"
            prompt += "The purpose of these examples is to provide complete, functional, and informative examples to non-experts, so that the code just-works, and they can modify it to suit their purposes if needed.\n"
            prompt += "\n"
            prompt += "Previously, you were provided with highly relevant files in the repository, and asked to generate a code example.\n"
            prompt += "That code example was executed, and the results are below.\n"
            prompt += "Now, your task is to reflect on the code example and it's output/results, and determine if it ran successfully, and if not, determine what went wrong and fix any errors.\n"
            prompt += "To assist with this task, the following will be provided below:\n"
            prompt += "- The repository description, which was automatically generated from the README file.\n"
            prompt += "- The highly relevant files previously identified in the repository\n"
            prompt += "- The example code you generated (including the code, requirements, and runscript)\n"
            prompt += "- The output/results of the code example, including any errors or issues that were encountered.\n"
            prompt += "- An assessment of some of the issues that may be wrong with this code.\n"
            prompt += "\n"

            prompt += "# Repository Description\n"
            prompt += "The repository URL is: " + repo_url + "\n"
            prompt += "\n"
            prompt += "A description of the repository (that was automatically generated from the README file) is as follows:\n"
            prompt += "```\n"
            prompt += repository_description + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# List of highly relevant files from the repository:\n"
            total_files = len(selected_files)
            prompt += "Here is a list of the " + str(total_files) + " most highly relevant files that you found in the repository (that will be provided below):\n"
            if (total_files == 0):
                prompt += "No files were classified as highly relevant.\n"
            else:
                file_idx = 1
                for file_info in selected_files:
                    # print("File info (file idx): " + str(file_idx))
                    # print(json.dumps(file_info, indent=4, ensure_ascii=False, default=lambda o: o.value if isinstance(o, enum.Enum) else o))
                    # print("\n")

                    prompt += "- File " + str(file_idx) + ": " + file_info["file_path_repo"]
                    prompt += "Metadata: " + json.dumps(file_info["response"]) + "\n"
                    prompt += "\n"
                    file_idx += 1

            prompt += "# File Contents\n"
            prompt += "Below is the contents of each of the highly relevant files.\n"
            prompt += "\n"
            for file_info in selected_files:
                # Get the filename (with path), to use as a key
                # Get the file contents
                try:
                    file_path = file_info.get("file_path_repo", None)
                    if (file_path is None):
                        continue
                    file_contents_str = file_contents.get(file_path, None)
                    if (file_contents_str is None):
                        print("ERROR: File contents not found for file: " + str(file_path))
                        continue
                    prompt += f"## File: {file_path}\n"
                    prompt += "Contents:\n"
                    prompt += "```\n"
                    prompt += file_contents_str + "\n"
                    prompt += "```\n"
                    prompt += "\n"
                except KeyError:
                    print("ERROR: File contents not found for file: " + str(file_path))
                    #file_contents_str = "File contents not found."

            prompt += "\n"

            prompt += "# Your previously generated code example\n"
            prompt += "Below is the example code that you generated, including the code, requirements, and runscript.\n"
            prompt += "\n"
            prompt += "## Code (main.py)\n"
            prompt += "```\n"
            prompt += current_example_state.get("code", "ERROR: No code found. This is a critical error!") + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## requirements.txt\n"
            prompt += "```\n"
            prompt += current_example_state.get("requirements", "ERROR: No requirements found. This is a critical error!") + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## run.sh\n"
            prompt += "```\n"
            prompt += current_example_state.get("runscript", "ERROR: No runscript found. This is a critical error!") + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# Change log\n"
            prompt += "Below is the change log (between debugging iterations) of the example code.\n"
            prompt += "The maximum number of debugging iterations is: " + str(self.max_debug_iterations) + "\n"
            prompt += "This can help you determine what issues you previously identified/fixes you tried, to avoid repeating actions that are ineffectual.\n"
            change_log = current_example_state.get("change_log", [])
            prompt += "```\n"
            if (len(change_log) == 0):
                prompt += "No changelog found. This is likely the first debugging iteration.\n"
            else:
                prompt += json.dumps(change_log, indent=4, ensure_ascii=False) + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "## POTENTIAL ISSUES WITH THIS CODE (IMPORTANT!)\n"
            prompt += "In a previous reflection step, the following potential issues were identified with this code.  You should direct your efforts to fixing them, and any other issues not mentioned here that you identify.\n"
            prompt += "Also, if you keep struggling with the same/similar issues, please consider backing-off to creating a less ambitious example that is still useful.\n"
            prompt += "```\n"
            prompt += json.dumps(error_information, indent=4) + "\n"
            prompt += "```\n"

            prompt += "\n"

            prompt += "# Output/results of the code example\n"
            prompt += "Below is the output/results of the code example you generated, including any errors or issues that were encountered.\n"
            prompt += "You will need to analyze this output/results to determine if the code example ran successfully, and if not, determine what went wrong and fix any errors.\n"
            prompt += "\n"

            # Token limit per log file
            MAX_TOKENS_PER_LOG_FILE = 10000
            # Adaptively adjust the max tokens per log file based on the total number of tokens in the prompt
            #MAX_TOKENS_FULL_PROMPT = 150000 - 5000  # 150k tokens total, minus a buffer for the rest of the prompt

            total_prompt_tokens = countTokens(prompt)
            tokens_left = MAX_TOKENS_FULL_PROMPT - total_prompt_tokens
            MAX_TOKENS_PER_LOG_FILE = min(MAX_TOKENS_PER_LOG_FILE, tokens_left // 8)  # Divide by 5 to leave room for other logs
            if (MAX_TOKENS_PER_LOG_FILE > 10000):
                # Clip back down to 10k tokens if the limit exceeds that.
                MAX_TOKENS_PER_LOG_FILE = 10000
            if (MAX_TOKENS_PER_LOG_FILE < 1000):
                # If the limit is too low, set it to 1k tokens
                print("WARNING: MAX_TOKENS_PER_LOG_FILE is very low (" + str(MAX_TOKENS_PER_LOG_FILE) + "), this may lead to highly truncated logs.")
                MAX_TOKENS_PER_LOG_FILE = 1000

            execution_results = current_example_state.get("execution_result", [])
            last_execution_result = execution_results[-1] if execution_results else {}
            # Python stdout/stderr
            prompt += "## Python stdout\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("python.stdout", "ERROR: No Python stdout found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("python.stdout", "ERROR: No Python stdout found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            python_stdout = last_execution_result.get("python.stdout", "ERROR: No Python stdout found.\n")
            if (python_stdout is None):
                python_stdout = "ERROR: No Python stdout found.\n"
            prompt += trimPromptComponentLog(python_stdout.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## Python stderr\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("python.stderr", "ERROR: No Python stderr found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("python.stderr", "ERROR: No Python stderr found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            python_stderr = last_execution_result.get("python.stderr", "ERROR: No Python stderr found.\n")
            if (python_stderr is None):
                python_stderr = "ERROR: No Python stderr found.\n"
            prompt += trimPromptComponentLog(python_stderr.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # pip stdout/stderr
            prompt += "## PIP stdout\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("pip.stdout", "ERROR: No PIP stdout found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("pip.stdout", "ERROR: No PIP stdout found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            pip_stdout = last_execution_result.get("pip.stdout", "ERROR: No PIP stdout found.\n")
            if (pip_stdout is None):
                pip_stdout = "ERROR: No PIP stdout found.\n"
            prompt += trimPromptComponentLog(pip_stdout.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## PIP stderr\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("pip.stderr", "ERROR: No PIP stderr found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("pip.stderr", "ERROR: No PIP stderr found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            pip_stderr = last_execution_result.get("pip.stderr", "ERROR: No PIP stderr found.\n")
            if (pip_stderr is None):
                pip_stderr = "ERROR: No PIP stderr found.\n"
            prompt += trimPromptComponentLog(pip_stderr.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Runscript stdout/stderr
            prompt += "## Run script stdout\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("runscript.stdout", "ERROR: No runscript stdout found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("runscript.stdout", "ERROR: No runscript stdout found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            runscript_stdout = last_execution_result.get("runscript.stdout", "ERROR: No runscript stdout found.\n")
            if (runscript_stdout is None):
                runscript_stdout = "ERROR: No runscript stdout found.\n"
            prompt += trimPromptComponentLog(runscript_stdout.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## Run script stderr\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("runscript.stderr", "ERROR: No runscript stderr found.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("runscript.stderr", "ERROR: No runscript stderr found.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            runscript_stderr = last_execution_result.get("runscript.stderr", "ERROR: No runscript stderr found.\n")
            if (runscript_stderr is None):
                runscript_stderr = "ERROR: No runscript stderr found.\n"
            prompt += trimPromptComponentLog(runscript_stderr.split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Log file
            prompt += "## Log file (log.json)\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("log", "ERROR: No log file found. This is a critical error.\n"))
            #prompt += trimPromptComponentLog(last_execution_result.get("log", "ERROR: No log file found. This is a critical error.\n").split("\n"), maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            log_file = last_execution_result.get("log", "ERROR: No log file found. This is a critical error.\n")
            if (log_file is None):
                log_file = "ERROR: No log file found. This is a critical error.\n"
            # Check the type
            if (isinstance(log_file, list) != True):
                # If it's not a list, convert it into something that can be split
                if (isinstance(log_file, dict)):
                    # If it's a dictionary, convert it to a string
                    log_file = json.dumps(log_file, indent=4, ensure_ascii=False)
                else:
                    log_file = str(log_file)
                log_file = log_file = log_file.split("\n")
            prompt += trimPromptComponentLog(log_file, maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Results file
            prompt += "## Results file (results.json)\n"
            prompt += "```\n"
            #prompt += str(current_example_state.get("results", "ERROR: No results file found. This is a critical error.\n"))
            #prompt += last_execution_result.get("results", "ERROR: No results file found. This is a critical error.\n")
            results_file = last_execution_result.get("results_json", "ERROR: No results file found. This is a critical error.\n")
            if (results_file is None):
                results_file = "ERROR: No results file found. This is a critical error.\n"
            # Check the type
            if (isinstance(results_file, list) != True):
                # If it's not a list, convert it into something that can be split
                if (isinstance(results_file, dict)):
                    # If it's a dictionary, convert it to a string
                    results_file = json.dumps(results_file, indent=4, ensure_ascii=False)
                else:
                    results_file = str(results_file)
                results_file = results_file.split("\n")
            prompt += trimPromptComponentLog(results_file, maxTokens=MAX_TOKENS_PER_LOG_FILE) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Container-specific errors
            prompt += "## Other container-specific errors (if any)\n"
            prompt += "```\n"
            prompt += str(last_execution_result.get("other_errors", "ERROR: No other errors found.\n"))
            prompt += "```\n"
            prompt += "\n"


            prompt += "# Why you are doing this\n"
            prompt += "You are part of an automated scientific discovery system that is working to rapidly accelerate the pace of scientific discovery, and generally make scientific discovery faster, less expensive, and more accessible. Performing well at this task could have large positive benefits for humanity.  Performing poorly at this task is considered a critical failure, and will (at a minimum) cost time, money, and other resources, but could also lead to incorrect results (a strongly negative outcome).\n"
            prompt += "\n"

            prompt += "# Container Environment\n"
            prompt += "You will be generating a complete, working code example that runs in a pre-existing container.  Here is more information about the container:\n"
            prompt += "- The container is running Ubuntu 22.04.\n"
            prompt += "- You need to install the required Python version using conda (e.g. `conda create -y -n myenv python=3.12`). Conda is not installed by default, use the example to install miniconda.\n"
            prompt += "- You need to install the required Python dependencies using pip (e.g. `pip install -r requirements.txt`) after activating the conda environment.\n"
            prompt += "- Very minimal system-level packages, so you may need to install them using `apt-get install`.\n"
            prompt += "- Runtime: Currently, the maximum runtime of the container is " + str(self.max_runtime_mins) + " minutes, so you should frame your code to run within that time limit. If that is absolutely not possible for a technical reason, set a flag called `impossible_to_run_within_limit` flag to be `TRUE` in the metadata output.\n"
            prompt += "- Runtime (2): It's important to note that " + str(self.max_runtime_mins) + " minutes is an absolute maximum -- you are STRONGLY encouraged to have code examples that run as quickly as possible, while still being straightforward.\n"
            prompt += "\n"
            prompt += "## Additional container information\n"
            prompt += "- The container is CPU-only, and DOES NOT have any GPU resources.\n"
            prompt += "- The container does have internet access\n"
            prompt += "- The container does NOT have any API keys (such as OPENAI API keys), so it can not successfully make calls that require API keys.\n"
            prompt += "\n"

            prompt += "# Logging\n"
            # prompt += "For demonstration and debugging purposes, your code must output the following output files in its working directory:\n"
            # prompt += "- `log.json`: A JSON file that contains a log of what the code did at each time step, to help understand what the code is doing, and debug any issues that arise.\n"
            # prompt += "- `results.json`: a JSON file that contains the final output/results of the code, to show that it worked successfully.\n"

            # prompt += "\n"
            prompt += "For demonstration and debugging purposes, your code must output the following output files in its working directory:\n"
            prompt += "- `log.json`: A JSON file that contains a log of what the code did at each time step, to help understand what the code is doing, and debug any issues that arise.\n"
            prompt += "- `results.json`: a JSON file that contains the final output/results of the code, to show that it worked successfully.\n"
            prompt += "- Anything you'd like to save to demontrate functionality, such as data, figures, etc., must be placed in a `to_save/` subdirectory off the working directory that `main.py` is in. The total size of these files should not be too large (e.g. ideally under 25MB total), or the download will timeout.\n"
            prompt += "These files should be saved often, rather than once at the end, so that a crash doesn't prevent you from gaining at least partial information about the code execution.\n"
            prompt += "Only the above files (`log.json`, `results.json`, and the contents of the `to_save/` directory) will be saved and returned to the user -- any other files created during runtime are ephemeral and will be deleted after execution, and not seen by the user.\n"
            prompt += "\n"

            prompt += "# Output format\n"
            prompt += "Now it is time for you to generate any debugging revisions to the code example, based on the output/results of the code example you generated.\n"
            prompt += "If no revisions are required, and the code is executing successfully, you should output the same code as before, but with the metadata updated to indicate that the code example was successful.\n"
            prompt += "You are encouraged to think and plan before writing your code/output, but the code/output (contained within exactly 4 codeblocks) must be the last thing you write/output.\n"
            prompt += "You will need to generate exactly 4 output files, each enclosed in codeblock backticks (```).  Do NOT use backticks in your generated files as this will likely not parse correctly and be a critical failure.\n"
            prompt += "The 4 output files, in order, are:\n"
            prompt += "- A program (that will be saved as `main.py`) that contains the example code\n"
            prompt += "- A requirements file (saved as 'requirements.txt') that lists the dependencies, that will be automatically installed using pip.\n"
            prompt += "- A runscript (saved as `run.sh`) that will perform any environment-specific setup in the Ubuntu 22.04 container (e.g. cloning the original repository if required, installing any system-level packages (e.g. `apt-get install graphviz`), etc.)\n"
            #prompt += "- A set of metadata in JSON format (saved as `metadata.json`) that contains information about the example). It must be a dictionary with the following keys: `repository_url`:str, `description`:str, `example_description`:str, `example_type`:str, `user_type`:str, `compute_requirements`:dict, `issues_identified`:list(str), and `changes_made`:list(str).  The `compute_requirements` key must be a dictionary with the following keys: `gpu`:bool, `min_gpu_memory_gb`:int, `cpu_cores`:int, `ram_gb`:int, `disk_space_gb`:int, `api_keys_required`:list[str], and `api_keys_optional`:list[str].  For unknown values, write none/null, not a blank string, or zero.\n"
            prompt += "- A set of metadata in JSON format (saved as `metadata.json`) that contains information about the example). It must be a dictionary with the following keys: `repository_url`:str, `description`:str, `example_description`:str, `example_inclusion_criteria`:str, `example_exclusion_criteria`:str, `example_type`:str, `user_type`:str, `compute_requirements`:dict, `issues_identified`:list(str), and `changes_made`:list(str).  The `compute_requirements` key must be a dictionary with the following keys: `gpu_required`:bool, `gpu_normally_recommended`:bool, `min_gpu_memory_gb`:int, `cpu_cores`:int, `ram_gb`:int, `disk_space_gb`:int, `api_keys_required`:list[str], `api_keys_optional`:list[str], and `time_constraint_limited`:bool.  For unknown values, write none/null, not a blank string, or zero.\n"
            prompt += "\n"
            prompt += "## Example Output\n"
            prompt += "Below is a cartoon sketch of the type of example code that you should generate.\n"
            prompt += "First codeblock (main.py):\n"
            prompt += "```\n"
            prompt += "# Imports here\n"
            prompt += "\n"
            prompt + "# Describe what example 1 does here\n"
            prompt += "def example1():\n"
            prompt += "    # Code for example 1 here\n"
            prompt += "\n"
            prompt += "# Describe what example 2 does here\n"
            prompt += "def example2():\n"
            prompt += "    # Code for example 2 here\n"
            prompt += "\n"
            prompt += "# etc.\n"
            prompt += "\n"
            prompt += "# Main entry point (to run the examples)\n"
            prompt += "if __name__ == '__main__':\n"
            prompt += "    example1()\n"
            prompt += "    example2()\n"
            prompt += "    # etc.\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "Second codeblock (requirements.txt):\n"
            prompt += "```\n"
            prompt += "# List of dependencies here, one per line\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "Third codeblock (run.sh):\n"
            prompt += "```\n"
            prompt += "#!/bin/bash\n"
            prompt += "# Run any environment-specific setup here.  Note, your runscript must ALWAYS have the REQUIRED elements.\n"
            prompt += "# Example is below\n"
            prompt += "# Step 1: (OPTIONAL) If the original repository needs to be cloned, do that here\n"
            prompt += "git clone " + repo_url + "  # Example \n"
            prompt += "# Step 2: (OPTIONAL) Install any system-level packages here\n"
            prompt += "apt-get update && apt-get install -y graphviz # This is just an example -- don't install graphviz unless you need it!\n"
            # prompt += "# Step 3: (REQUIRED) Create the conda environment\n"
            # prompt += "conda create -y -n myenv python=3.12  # Select appropriate Python version\n"
            # prompt += "# Step 4: (REQUIRED) Activate the conda environment\n"
            # prompt += "source activate myenv\n"
            # prompt += "# Step 5: (REQUIRED) Install the Python dependencies\n"
            # prompt += "export PIP_ROOT_USER_ACTION=ignore\n"
            # prompt += "pip install -r requirements.txt\n"
            prompt += """# Step 3 (required): Minimal system packages for downloading Miniconda
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget bzip2 ca-certificates
rm -rf /var/lib/apt/lists/*

# Step 4 (required): Install Miniconda (if not already)
if ! command -v conda >/dev/null 2>&1; then
  echo "[RUN.SH] Installing Miniconda..."
  CONDA_DIR="/root/miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniconda.sh
  export PATH="$CONDA_DIR/bin:$PATH"
else
  echo "[RUN.SH] Conda already available."
  CONDA_DIR="$(dirname "$(dirname "$(command -v conda)")")"
  export PATH="$CONDA_DIR/bin:$PATH"
fi

# Initialize conda in this shell
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Step 5 (required): Configure conda to use conda-forge only (avoid Anaconda TOS issues)
echo "[RUN.SH] Configuring conda channels to use conda-forge only..."
conda config --system --set auto_update_conda false || true
conda config --system --set channel_priority strict || true
# Remove defaults/anaconda channels if present
conda config --system --remove channels defaults || true
conda config --system --remove-key default_channels || true
# Add conda-forge
conda config --system --add channels conda-forge || true

# Optional: attempt to accept TOS if the conda version supports it (best-effort)
if conda --help | grep -q "tos"; then
  echo "[RUN.SH] Attempting to accept Anaconda TOS (best-effort)..."
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

# Step 6 (required): Create the conda environment
ENV_NAME="myenv"
PY_VER="3.10"
if conda env list | grep -qE "^\\s*${ENV_NAME}\\s"; then
  echo "[RUN.SH] Conda env '${ENV_NAME}' already exists."
else
  echo "[RUN.SH] Creating conda env '${ENV_NAME}' with Python ${PY_VER} (from conda-forge)..."
  conda create -y -n "${ENV_NAME}" -c conda-forge python="${PY_VER}"
fi

# Step 7 (required): Activate env and install Python dependencies with pip
conda activate "${ENV_NAME}"
export PIP_ROOT_USER_ACTION=ignore

python -c "import sys; print('[RUN.SH] Python', sys.version)"
pip --version

echo "[RUN.SH] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

"""

            prompt += "# Step 8: Any other setup steps here\n"
            prompt += "# NOTE: Do *NOT* run the main program from here, that will be done separately.\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "Fourth codeblock (metadata.json):\n"
            prompt += "```\n"
            prompt += "{\n"
            prompt += "  \"repository_url\": \"" + repo_url + "\",\n"
            prompt += "  \"description\": \"" + repository_description + "\",\n"
            prompt += "  \"example_description\": \"This is a complete, working code example that...\", # A concise, information-dense, description of what what this code example demonstrates how to do, which would be useful both for humans to read but also search to lock onto important keywords or terms.\n"
            prompt += "  \"example_inclusion_criteria\": \"...\",  # A concise, information-dense description of the broad situations where this example would be useful to a user, perhaps solving particular kinds of tasks, and/or working with particular kinds of data, etc.\n"
            prompt += "  \"example_exclusion_criteria\": \"...\",  # A concise, information-dense description of the broad situations where this example would NOT be useful to a user, perhaps situations where it would fail, or not be applicable, etc.\n"
            prompt += "  \"example_type:\": \"...\" # One of exactly the following 4 keys: `library_demo` (showing how to use libraries, like transformers), `application_demo` (showing how to use complete standalone programs, like dot/graphviz), `code_snippet_demo` (showing how to use complete self-contained code), or `other` (all other).\n"
            prompt += "  \"compute_requirements\": { # Use exactly these keys.  For unknown values, write none/null.\n"
            prompt += "    \"gpu_required\": true, # bool\n"
            prompt += "    \"gpu_normally_recommended\": true, # Set to `true` if the repository normally suggests/requires using a GPU, regardless of whether this code example requires one.\n"
            prompt += "    \"min_gpu_memory_gb\": 16, # int\n"
            prompt += "    \"cpu_cores\": 8, # int\n"
            prompt += "    \"ram_gb\": 32, # int \n"
            prompt += "    \"disk_space_gb\": 100, # int\n"
            prompt += "    \"api_keys_required\": [\"OPENAI_API_KEY\"], # list of str, of any API key *normally required* for primary functionality of the repository, regardless of whether required here.\n"
            prompt += "    \"api_keys_optional\": [\"SEMANTICSCHOLAR_API_KEY\"] # list of str, of any API key *optional* for primary functionality (i.e. Semantic Scholar keys are optional because it still allows access with no key, just at a slower rate limit),\n"
            prompt += "    \"time_constraint_limited\": false, # bool: Does this repository normally require more time to run than is available in the container (currently " + str(self.max_runtime_mins) + " minutes), and due to this, a limited version was run? Would having more time enable significantly better/more accurate results? If so, `true`, otherwise, `false`.\n"
            prompt += "  },\n"
            prompt += "  \"issues_identified\": [] # list of strings, each concisely describing any issues that were identified with the code example.\n"
            prompt += "  \"changes_made\": [] # list of strings, each concisely describing any changes that were made to the code example during this round of reflection to fix the above issues.\n"
            prompt += "}\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "# Important Notes\n"
            prompt += "- You must return exactly 4 codeblocks, in the order specified above.\n"
            prompt += "- You are encouraged to think and plan before writing your code/output, but the code/output (contained within exactly 4 codeblocks) must be the last thing you write/output, or it will not be parsed correctly.\n"
            prompt += "- You must not use backticks in your generated files, as this will likely not parse correctly and be a critical failure.\n"
            prompt += "- The code must be complete, functional, work on the first try in a Ubuntu 22.04 container, informative, and demonstrate how to use the code in the repository to perform its primary function.\n"
            prompt += "- The examples must NOT be trivial, they must be useful, clear demonstrations of utility.\n"
            prompt += "- The examples should ideally illustrate the main function of the repository -- the things a normal user is likely to do with it -- and not focus on edge cases or unusual functionality.\n"
            prompt += "- The top-level function names of the examples must be `example1()`, `example2()`, `example3()`, etc.  The main entry point of the code must be `if __name__ == '__main__':`.\n"
            prompt += "- The examples are intended to be USEFUL and MODIFIABLE by modifying the code example itself. If you just output a function that calls an existing demo or example, that is not useful or acceptable, as it will not be easily modifiable. You're trying to show a non-expert how to use this code, in a way that they can easily modify to use for themselves.\n"
            prompt += "- Please remember the overall goal of this task is to generate complete, working examples.  If, when debugging, a complex example doesn't appear to be working, you should consider backing-off to a simpler example that's still useful.\n"
            prompt += "- If the code could reasonably be expected to require API keys, list them int he `api_keys_required` or `api_keys_optional` keys in the metadata, as appropriate.  This is very important, otherwise we won't know what code we can run in the container, and will waste time and money trying to run code that can't run.\n"
            prompt += "- Your code should VERBOSELY output what it is doing, so that a non-expert can understand what is happening, but also debug it if something is not working as expected.  It should also provide an EASY, OBVIOUS mechanism to verify that the code is successfully accomplishing it's goal correctly and faithfully.\n"
            prompt += "- There should be very detailed (but concise, information-dense) documentation at the top of the main.py file, describing exactly what input the code expects (with examples), what output it produces (with examples), and any other relevant information to getting it to run for a non-expert.\n"
            prompt += "- All the codeblocks must be properly terminated (``` at the start and end).  Any JSON must be valid and properly formatted JSON.  Your output will be automatically parsed, and any deviations in the output format will cause a crash, which is a critical error.\n"
            prompt += "- The final line in your output should be a triple backtick (```), indicating the end of the last codeblock (the metadata).\n"
            prompt += "- All your information must be accurate. Do not hallucinate.\n"

            return prompt



        # Step 1: Pick out the most highly relevant files from the sorted file classifications, up to the maximum number of tokens.
        MAX_TOKENS = 100000     # 100k tokens total (across all files)
        # Flatten the list of files
        #all_files = selected_files
        all_files = current_example_state.get("classification_list", [])
        # Remove any files that don't have a relevance rating (also filter out any that are None)
        #all_files = [file for file in all_files if "how_relevant_rating" in file["response"]]
        all_files = [file for file in all_files if file.get("response", {}).get("how_relevant_rating", None) is not None]
        # Sort the files by how relevant rating, descending
        all_files.sort(key=lambda x: x["response"]["how_relevant_rating"], reverse=True)
        # Pick out the most highly relevant files, up to the maximum number of tokens
        selected_files = []
        total_tokens = 0
        file_content = {}
        TRUNCATED_SIZE = 10000  # If we truncate a file, truncate it to 10k tokens (to give room for other files in the context)
        for file in all_files:
            #print("Parsing file:")
            #print(json.dumps(file, indent=4, ensure_ascii=False))
            # Use encoder for Enum
            #print(json.dumps(file, indent=4, default=lambda o: o.value if isinstance(o, enum.Enum) else o))
            #print("\n")

            # First, try to load the file, to place it in the file content dictionary
            file_contents = None
            file_rating = file["response"]["how_relevant_rating"]
            # If the rating is less than 3, skip it
            if (file_rating < 3):
                #print(f"Skipping file {file['file_path_repo']} due to low relevance rating ({file_rating})")
                continue

            try:
                file_path_local = file["file_path_local"]
                file_path_repo = file["file_path_repo"]
                with open(file_path_local, 'r', encoding='utf-8') as f:
                    file_contents = f.read()
                file_content[file_path_repo] = file_contents
            except Exception as e:
                print(f"Error reading file {file_path_local}: {e}")

            if (file_contents is None):
                print(f"Error: File contents not found for file {file_path_local}")
                continue    # Do not continue with this file, since we can't use it

            # Add the file contents to the file content dictionary
            file_tokens = file["file_num_tokens"]
            # Check how many tokens are remaining
            remaining_tokens = MAX_TOKENS - total_tokens
            if (file_tokens <= remaining_tokens):
                # Add the file to the selected files
                selected_files.append(file)
                total_tokens += file_tokens
            # Check if there's enough space for a truncated version
            elif (remaining_tokens >= TRUNCATED_SIZE):
                # Truncate the file contents to the remaining tokens
                file_contents = trimToMaxTokens(file_contents, remaining_tokens)
                # Add the file to the selected files, with the truncated contents
                file_content[file_path_repo] = file_contents + "\n### NOTE: THIS LONG FILE WAS TRUNCATED TO FIT WITHIN THE AVAILABLE SPACE ###\n"
                selected_files.append(file)
                total_tokens += TRUNCATED_SIZE
            else:
                # Not enough space for this file, skip it
                #print(f"Skipping file {file['file_path_repo']} due to insufficient space (remaining tokens: {remaining_tokens}, file tokens: {file_tokens})")
                continue


        # Step 2A: Determine the maximum number of tokens for the code generation model
        MAX_TOKENS_FULL_PROMPT = 150000 - 5000  # 150k tokens total, minus a buffer for the rest of the prompt
        if ("gpt-oss-120b" in model_str):
            MAX_TOKENS_FULL_PROMPT = 95000 - 5000       # ~130k total tokens, but that includes the response as well.

        # Step 2: Generate the prompt to generate the initial example
        repo_url = current_example_state.get("repo_url", "No repository URL provided.")
        repository_description = current_example_state.get("repository_description", "No repository description provided.")
        #selected_files = current_example_state.get("classification_list", [])
        # def mkPromptErrorDetection(repository_description:str, repo_url:str, selected_files:list, file_contents:dict, current_example_state:dict):
        prompt = mkPromptErrorDetection(repository_description=repository_description, repo_url=repo_url, selected_files=selected_files, file_contents=file_content, current_example_state=current_example_state, MAX_TOKENS_FULL_PROMPT=MAX_TOKENS_FULL_PROMPT)

        # Make sure the prompt isn't too long for the model
        if (countTokens(prompt) > MAX_TOKENS_FULL_PROMPT):
            print(f"Error: The prompt is too long ({countTokens(prompt)} tokens), exceeding the maximum of {MAX_TOKENS_FULL_PROMPT} tokens. This is a critical error.")
            return {
                "error": "Prompt too long for model (" + model_str + "): " + str(countTokens(prompt)) + " tokens, max is " + str(MAX_TOKENS_FULL_PROMPT) + " tokens."
            }

        # Step 3: Call the language model to generate the initial example
        total_cost = 0.0
        #model_str_code_generation = "o3-mini"
        #model_str_code_generation = "o4-mini"
        responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.0, jsonOut=False)
        total_cost += cost

        # Get the 'error detection' response. Check that it at least has the `is_finished` key.
        if (responseJSON == None) or (isinstance(responseJSON, dict) != True) or ("is_finished" not in responseJSON):
            # Some error has happened, the output is not as expected
            print("ERROR: The response from the language model during the error detection phase was not as expected, and does not appear to be the JSON response expected.\n")
            return None


        error_information = responseJSON

        # Check to see if the task is finished
        is_finished = error_information.get("is_finished", False)
        if (is_finished == True):
            # If 'is_finished' is marked as true, then the debugging process is completed.  Finish here.
            # Return
            packed = {
                "repo_url": repo_url,
                "repository_description": repository_description,
                "model": model_str,
                "total_cost": total_cost,
                "is_finished": is_finished,
                "error_information": error_information,
            }
            return packed



        # Step 3B: If we reach here, the code was not determined to be finished.  Reflect the code.
        #def mkPromptCodeReflection(repository_description:str, repo_url:str, selected_files:list, file_contents:dict, current_example_state:dict, error_information:list):
        prompt = mkPromptCodeReflection(repository_description=repository_description, repo_url=repo_url, selected_files=selected_files, file_contents=file_content, current_example_state=current_example_state, error_information=error_information, MAX_TOKENS_FULL_PROMPT=MAX_TOKENS_FULL_PROMPT)
        # Make sure the prompt isn't too long for the model
        if (countTokens(prompt) > MAX_TOKENS_FULL_PROMPT):
            print(f"Error: The prompt is too long ({countTokens(prompt)} tokens), exceeding the maximum of {MAX_TOKENS_FULL_PROMPT} tokens. This is a critical error.")
            return {
                "error": "Prompt too long for model (" + model_str + "): " + str(countTokens(prompt)) + " tokens, max is " + str(MAX_TOKENS_FULL_PROMPT) + " tokens."
            }

        responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.0, jsonOut=False)
        total_cost += cost

        # Step 4: Parse the response (the text) into 4 codeblocks
        if (responseText is None):
            print("Error: No response from the language model for generating the initial example")
            return None
        codeblocks = find_codeblocks(responseText)
        print("Returned " + str(len(codeblocks)) + " codeblocks")

        if (len(codeblocks) != 4):
            print("Error: Expected 4 codeblocks, but got " + str(len(codeblocks)) + ".  This is a critical failure.")
            # Retry with slightly higher temperature
            prompt += "\n\nIMPORTANT NOTE: The previous response did not follow the instructions correctly, and did not return exactly 4 codeblocks.  Please try again, and ensure that you return exactly 4 codeblocks, in the order specified, and do not use backticks in your generated files, as this will likely not parse correctly and be a critical failure.\n"
            responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=32000, temperature=0.1, jsonOut=False)
            total_cost += cost
            if (responseText is None):
                print("Error: No response from the language model for generating the initial example (retry)")
                return None
            codeblocks = find_codeblocks(responseText)
            print("Returned " + str(len(codeblocks)) + " codeblocks after retry")

        if (len(codeblocks) != 4):
            print("Error: Still expected 4 codeblocks, but got " + str(len(codeblocks)) + ".  This is a critical failure.")
            # Feed back into a simpler model to try to fix it
            return None

        code = codeblocks[0]
        requirements = codeblocks[1]
        runscript = codeblocks[2]
        metadata = codeblocks[3]
        # Try to parse the metadata JSON
        metadata_json = None
        # try:
        #     metadata_json = json.loads(metadata)
        # except json.JSONDecodeError as e:
        #     print(f"Error parsing metadata JSON: {e}")
        #     # Feed back into a simpler model to try to fix it
        #     return None
        metadata_json, cost_fix = parse_and_fix_json(metadata, model_str="gpt-5-mini")
        total_cost += cost_fix

        # Store the total cost of this generation
        self.costs["generate_initial_code_example"] = total_cost

        # Return
        packed = {
            "repo_url": repo_url,
            "repository_description": repository_description,
            "model": model_str,
            "total_cost": total_cost,
            "is_finished": False,       # If we're still debugging, the example isn't finished.
            "error_information": error_information,
            "code": code,
            "requirements": requirements,
            "runscript": runscript,
            "metadata": metadata_json,
        }
        return packed

import json
import html
import os

# Convert the JSON log file into a self-contained HTML report. Generated w/GPT-5.
def convert_log_to_html_report(
    filename_current_state_json: str,
    filename_output_html: str,
    highlight_js_path: str | None = None,
    highlight_css_path: str | None = None,
):
    # Step 1: Try to load the current state
    try:
        with open(filename_current_state_json, 'r', encoding='utf-8') as f:
            current_state = json.load(f)
    except Exception as e:
        print(f"Error loading current state JSON file {filename_current_state_json}: {e}")
        print("Cannot generate HTML report.")
        return

    # Step 2: Extract critical aspects of the report

    # Metadata
    metadata = current_state.get("metadata", {})
    error_information = current_state.get("error_information", {})
    if (error_information is None):
        error_information = {}
    repo_url = current_state.get("repo_url", None)
    repository_description = metadata.get("description", None)
    example_description = metadata.get("example_description", None)
    example_inclusion_criteria = metadata.get("example_inclusion_criteria", None)
    example_exclusion_criteria = metadata.get("example_exclusion_criteria", None)
    correct_behavior = error_information.get("correct_behavior", None)
    critical_issues = error_information.get("critical_issues", [])
    issues = error_information.get("issues", [])
    is_faithful = error_information.get("is_faithful", None)
    is_accurate = error_information.get("is_accurate", None)
    is_complete = error_information.get("is_complete", None)
    fac_statement = error_information.get(
        "faithfulness_accurate_completeness_statement", None
    )
    status = current_state.get("status", None)
    status_message = current_state.get("status_message", None)
    model_str_code = current_state.get("model_str_code", None)
    model_str_fast = current_state.get("model_str_fast", None)
    total_cost = current_state.get("total_cost", 0.0)
    costs = current_state.get("costs", {})
    total_runtime_secs = current_state.get("total_runtime_secs", 0.0)
    runtimes = current_state.get("runtimes", {})
    max_runtime_mins = current_state.get("max_runtime_mins", None)
    max_debug_iterations = current_state.get("max_debug_iterations", None)
    is_finished = current_state.get("is_finished", False)

    # Change log
    change_log = current_state.get("change_log", [])

    # Code
    code = current_state.get("code", None)
    requirements = current_state.get("requirements", None)
    runscript = current_state.get("runscript", None)

    # File classification list
    classification_list = current_state.get("classification_list", [])

    # Execution Result
    execution_result = current_state.get("execution_result", [])
    last_execution_result = execution_result[-1] if len(execution_result) > 0 else {}
    result = last_execution_result.get("result", {})
    result_output = result.get("output", {})
    result_python_stdout = result_output.get("python.stdout", "No stdout captured.")
    result_python_stderr = result_output.get("python.stderr", "No stderr captured.")
    result_log = result_output.get("log", {})
    result_results_json = result_output.get("results_json", {})
    files_downloaded = result_output.get("files_downloaded", {})  # dict: {rel_path: size}
    file_path = result_output.get("file_path", "")  # base path for downloaded files

    # File path: We need to trim everything before 'modal-python-"
    if file_path is not None:
        modal_index = file_path.find("modal-python-")
        if modal_index >= 0:
            file_path = file_path[modal_index:]

    # Ensure we always have a dict for files_downloaded
    if not isinstance(files_downloaded, dict):
        files_downloaded = {}

    # Helpers
    def esc(value):
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return html.escape(value, quote=False)

    def esc_json(obj):
        try:
            text = json.dumps(obj, indent=2, sort_keys=True)
        except TypeError:
            text = json.dumps(str(obj), indent=2)
        return html.escape(text, quote=False)

    def bool_to_str(v):
        if v is True:
            return "Yes"
        if v is False:
            return "No"
        return "Unknown"

    def read_if_exists(path):
        if path is None:
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")
            return ""

    def html_list(items):
        """Return a clean <ul> list for items."""
        if not items:
            return "<span class='empty-note'>None</span>"
        out = "<ul>"
        for item in items:
            out += f"<li>{html.escape(str(item), quote=False)}</li>"
        out += "</ul>"
        return out

    # Generated output: identify "to-save" files and whether they look like images
    def is_image_path(path: str) -> bool:
        lower = path.lower()
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"):
            if lower.endswith(ext):
                return True
        return False

    def get_to_save_files(files_downloaded_dict, base_path=""):
        """
        files_downloaded_dict: {relative_path: size}
        base_path: directory to prepend when making links
        """
        results = []
        if not isinstance(files_downloaded_dict, dict):
            return results
        for rel_path in files_downloaded_dict.keys():
            # Be generous about spelling: 'to-save' or 'to_save'
            if "to-save" in rel_path or "to_save" in rel_path:
                full_path = os.path.join(base_path, rel_path) if base_path else rel_path
                results.append(
                    {
                        "name": rel_path,           # what we show to the user
                        "path": full_path,          # what we use in href/src
                        "note": "",
                        "is_image": is_image_path(rel_path),
                    }
                )
        return results

    to_save_files = get_to_save_files(files_downloaded, file_path)

    # Load highlight.js script and CSS (if provided)
    hljs_js = read_if_exists(highlight_js_path)
    hljs_css = read_if_exists(highlight_css_path)

    # === HTML generation ===
    html_out = ""

    # Head
    html_out += "<!DOCTYPE html>\n"
    html_out += "<html lang='en'>\n"
    html_out += "<head>\n"
    html_out += "  <meta charset='UTF-8'>\n"
    html_out += "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    title_str = f"CodeDistiller Log Report - {filename_current_state_json}"
    html_out += f"  <title>{esc(title_str)}</title>\n"
    html_out += "  <style>\n"

    # Base styles (adapted from COMMON_STYLES)
    html_out += "body {\n"
    html_out += "    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;\n"
    html_out += "    line-height: 1.6;\n"
    html_out += "    color: #333;\n"
    html_out += "    max-width: 1200px;\n"
    html_out += "    margin: 0 auto;\n"
    html_out += "    padding: 20px;\n"
    html_out += "    background-color: #f5f5f5;\n"
    html_out += "}\n\n"

    html_out += ".header {\n"
    html_out += "    display: flex;\n"
    html_out += "    align-items: center;\n"
    html_out += "    width: 100%;\n"
    html_out += "    height: 50px;\n"
    html_out += "    margin-bottom: 30px;\n"
    html_out += "    background-color: white;\n"
    html_out += "    padding: 10px 20px;\n"
    html_out += "    border-radius: 5px;\n"
    html_out += "    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n"
    html_out += "}\n\n"

    html_out += ".header a {\n"
    html_out += "    text-decoration: none;\n"
    html_out += "    color: #333;\n"
    html_out += "    font-size: 20px;\n"
    html_out += "    font-weight: bold;\n"
    html_out += "}\n\n"

    html_out += ".content {\n"
    html_out += "    background-color: white;\n"
    html_out += "    padding: 30px;\n"
    html_out += "    border-radius: 5px;\n"
    html_out += "    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n"
    html_out += "}\n\n"

    html_out += "h1 {\n"
    html_out += "    color: #2c3e50;\n"
    html_out += "    border-bottom: 2px solid #3498db;\n"
    html_out += "    padding-bottom: 10px;\n"
    html_out += "    margin-top: 0;\n"
    html_out += "}\n\n"

    html_out += "h2 {\n"
    html_out += "    color: #34495e;\n"
    html_out += "    margin-top: 30px;\n"
    html_out += "    margin-bottom: 15px;\n"
    html_out += "    font-size: 1.5em;\n"
    html_out += "}\n\n"

    html_out += "h3 {\n"
    html_out += "    color: #7f8c8d;\n"
    html_out += "    margin-top: 20px;\n"
    html_out += "    margin-bottom: 10px;\n"
    html_out += "    font-size: 1.1em;\n"
    html_out += "}\n\n"

    html_out += ".btn {\n"
    html_out += "    display: inline-block;\n"
    html_out += "    padding: 6px 12px;\n"
    html_out += "    background-color: #007bff;\n"
    html_out += "    color: white;\n"
    html_out += "    text-decoration: none;\n"
    html_out += "    border-radius: 4px;\n"
    html_out += "    font-size: 13px;\n"
    html_out += "    transition: background-color 0.3s;\n"
    html_out += "}\n\n"

    html_out += ".btn:hover {\n"
    html_out += "    background-color: #0056b3;\n"
    html_out += "}\n\n"

    html_out += "strong { color: #2c3e50; }\n"
    html_out += "p { margin: 10px 0; }\n"
    html_out += "ul, ol { margin: 10px 0; padding-left: 30px; }\n"
    html_out += "li { margin: 6px 0; }\n"
    html_out += "a { color: #3498db; text-decoration: none; }\n"
    html_out += "a:hover { text-decoration: underline; }\n\n"

    html_out += ".info-section {\n"
    html_out += "    background-color: #f8f9fa;\n"
    html_out += "    padding: 15px;\n"
    html_out += "    border-left: 4px solid #3498db;\n"
    html_out += "    margin: 15px 0;\n"
    html_out += "}\n\n"

    html_out += ".empty-note {\n"
    html_out += "    font-style: italic;\n"
    html_out += "    color: #7f8c8d;\n"
    html_out += "}\n\n"

    html_out += "pre {\n"
    html_out += "    background-color: #f4f4f4;\n"
    html_out += "    border: 1px solid #ddd;\n"
    html_out += "    border-radius: 4px;\n"
    html_out += "    padding: 12px;\n"
    html_out += "    overflow-x: auto;\n"
    html_out += "    font-size: 0.9em;\n"
    html_out += "}\n\n"

    html_out += "code {\n"
    html_out += "    font-family: 'Courier New', monospace;\n"
    html_out += "}\n\n"

    html_out += ".section { margin-bottom: 30px; }\n\n"

    html_out += "table {\n"
    html_out += "    width: 100%;\n"
    html_out += "    border-collapse: collapse;\n"
    html_out += "    margin: 20px 0;\n"
    html_out += "}\n\n"

    html_out += "th, td {\n"
    html_out += "    padding: 10px;\n"
    html_out += "    text-align: left;\n"
    html_out += "    border-bottom: 1px solid #ddd;\n"
    html_out += "    vertical-align: top;\n"
    html_out += "}\n\n"

    html_out += "th {\n"
    html_out += "    background-color: #f8f9fa;\n"
    html_out += "    font-weight: bold;\n"
    html_out += "    color: #2c3e50;\n"
    html_out += "}\n\n"

    html_out += "tr:hover { background-color: #f8f9fa; }\n\n"

    html_out += ".index-nav {\n"
    html_out += "    background-color: #f8f9fa;\n"
    html_out += "    padding: 15px;\n"
    html_out += "    border-radius: 4px;\n"
    html_out += "    border-left: 4px solid #3498db;\n"
    html_out += "}\n\n"

    html_out += ".back-to-top { margin-top: 10px; }\n\n"

    html_out += ".status-pill {\n"
    html_out += "    display: inline-block;\n"
    html_out += "    padding: 3px 8px;\n"
    html_out += "    border-radius: 12px;\n"
    html_out += "    font-size: 0.8em;\n"
    html_out += "    margin-left: 6px;\n"
    html_out += "}\n\n"

    html_out += ".status-success { background-color: #e8f5e9; color: #2e7d32; }\n"
    html_out += ".status-error { background-color: #ffebee; color: #c62828; }\n"
    html_out += ".status-pending { background-color: #fff8e1; color: #ff8f00; }\n\n"

    html_out += ".monosmall { font-family: 'Courier New', monospace; font-size: 0.85em; }\n\n"

    # Generated output previews
    html_out += ".output-preview {\n"
    html_out += "    margin-top: 8px;\n"
    html_out += "}\n\n"

    html_out += ".preview-img {\n"
    html_out += "    max-width: 500px;\n"
    html_out += "    max-height: 500px;\n"
    html_out += "    display: block;\n"
    html_out += "    margin-top: 8px;\n"
    html_out += "    border: 1px solid #ddd;\n"
    html_out += "    border-radius: 4px;\n"
    html_out += "}\n\n"

    # Inline highlight.js theme CSS if available
    if hljs_css:
        html_out += "/* highlight.js theme */\n"
        html_out += hljs_css
        html_out += "\n"

    html_out += "  </style>\n"

    # Inline highlight.js script if available
    if hljs_js:
        html_out += "  <script>\n"
        html_out += hljs_js
        html_out += "\n  </script>\n"

        # Auto-highlight script
        html_out += "  <script>\n"
        html_out += "  document.addEventListener('DOMContentLoaded', function() {\n"
        html_out += "    if (window.hljs) {\n"
        html_out += "      document.querySelectorAll('pre code').forEach(function(block) {\n"
        html_out += "        hljs.highlightElement(block);\n"
        html_out += "      });\n"
        html_out += "    }\n"
        html_out += "  });\n"
        html_out += "  </script>\n"

    html_out += "</head>\n"

    # Body + header
    html_out += "<body>\n"
    html_out += "  <a id='top'></a>\n"
    html_out += "  <div class='header'>\n"
    html_out += "    <a href='#top'>CodeDistiller Generation Report</a>\n"
    html_out += "  </div>\n"
    html_out += "  <div class='content'>\n"

    # Title and status
    html_out += "    <h1>Table of Contents</h1>\n"
    html_out += "    <p>\n"
    if repo_url:
        html_out += f"      <strong>Repository:</strong> <a href='{esc(repo_url)}' target='_blank'>{esc(repo_url)}</a>\n"
    else:
        html_out += "      <strong>Repository:</strong> Unknown\n"
    html_out += "    </p>\n"

    status_label = "Unknown"
    status_class = "status-pending"
    if isinstance(status, str):
        s = status.lower()
        if s in {"success", "ok", "completed"}:
            status_label = status
            status_class = "status-success"
        elif s in {"error", "failed", "failure"}:
            status_label = status
            status_class = "status-error"
        else:
            status_label = status
    elif status is not None:
        status_label = str(status)

    html_out += "    <p>\n"
    html_out += "      <strong>Run Status:</strong>\n"
    html_out += f"      <span class='status-pill {status_class}'>{esc(status_label)}</span>\n"
    if status_message:
        html_out += f"      <span style='margin-left:8px;' class='monosmall'>{esc(status_message)}</span>\n"
    html_out += "    </p>\n"

    # Index
    html_out += "    <div class='section index-nav'>\n"
    #html_out += "      <h2>Index</h2>\n"
    html_out += "      <ol>\n"
    html_out += "        <li><a href='#sec-overview'>Overview &amp; Metadata</a></li>\n"
    html_out += "        <li><a href='#sec-config'>Configuration &amp; Models</a></li>\n"
    html_out += "        <li><a href='#sec-error'>Error Assessment</a></li>\n"
    html_out += "        <li><a href='#sec-classification'>File Classification</a></li>\n"
    html_out += "        <li><a href='#sec-code'>Code &amp; Scripts</a></li>\n"
    html_out += "        <li><a href='#sec-execution'>Execution Result</a></li>\n"
    html_out += "        <li><a href='#sec-output'>Generated Output</a></li>\n"
    html_out += "        <li><a href='#sec-costs'>Cost &amp; Runtime Summary</a></li>\n"
    html_out += "        <li><a href='#sec-changelog'>Debugging Change Log</a></li>\n"
    html_out += "      </ol>\n"
    html_out += "    </div>\n"

    # Section 1: Overview & Metadata
    html_out += "    <div class='section' id='sec-overview'>\n"
    html_out += "      <h2>1. Overview &amp; Metadata</h2>\n"
    html_out += "      <div class='info-section'>\n"
    if repo_url:
        html_out += f"        <p><strong>Repository URL:</strong> <a href='{esc(repo_url)}' target='_blank'>{esc(repo_url)}</a></p>\n"
    if repository_description:
        html_out += f"        <p><strong>Repository Description:</strong> {esc(repository_description)}</p>\n"
    if example_description:
        html_out += f"        <p><strong>Example Description:</strong> {esc(example_description)}</p>\n"
    if example_inclusion_criteria:
        html_out += f"        <p><strong>Inclusion Criteria:</strong> {esc(example_inclusion_criteria)}</p>\n"
    if example_exclusion_criteria:
        html_out += f"        <p><strong>Exclusion Criteria:</strong> {esc(example_exclusion_criteria)}</p>\n"
    if not any(
        [
            repo_url,
            repository_description,
            example_description,
            example_inclusion_criteria,
            example_exclusion_criteria,
        ]
    ):
        html_out += "        <p class='empty-note'>No metadata available.</p>\n"
    html_out += "      </div>\n"
    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 2: Configuration & Models
    html_out += "    <div class='section' id='sec-config'>\n"
    html_out += "      <h2>2. Configuration &amp; Models</h2>\n"
    html_out += "      <table>\n"
    html_out += "        <tbody>\n"
    if model_str_code:
        html_out += f"          <tr><th>Code Model</th><td class='monosmall'>{esc(model_str_code)}</td></tr>\n"
    if model_str_fast:
        html_out += f"          <tr><th>Fast Model</th><td class='monosmall'>{esc(model_str_fast)}</td></tr>\n"
    if max_runtime_mins is not None:
        html_out += f"          <tr><th>Max Runtime (mins)</th><td>{esc(max_runtime_mins)}</td></tr>\n"
    if max_debug_iterations is not None:
        html_out += f"          <tr><th>Max Debug Iterations</th><td>{esc(max_debug_iterations)}</td></tr>\n"
    html_out += f"          <tr><th>Finished</th><td>{esc(bool_to_str(is_finished))}</td></tr>\n"
    html_out += "        </tbody>\n"
    html_out += "      </table>\n"
    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 3: Error Assessment
    html_out += "    <div class='section' id='sec-error'>\n"
    html_out += "      <h2>3. Error Assessment</h2>\n"
    html_out += "      <table>\n"
    html_out += "        <tbody>\n"

    html_out += f"          <tr><th>Faithful to Repository</th><td>{esc(bool_to_str(is_faithful))}</td></tr>\n"
    html_out += f"          <tr><th>Accurate</th><td>{esc(bool_to_str(is_accurate))}</td></tr>\n"
    html_out += f"          <tr><th>Complete</th><td>{esc(bool_to_str(is_complete))}</td></tr>\n"

    if fac_statement:
        html_out += f"          <tr><th>Summary Statement</th><td>{esc(fac_statement)}</td></tr>\n"

    if correct_behavior:
        if isinstance(correct_behavior, list):
            html_out += f"          <tr><th>Correct Behavior</th><td>{html_list(correct_behavior)}</td></tr>\n"
        else:
            html_out += f"          <tr><th>Correct Behavior</th><td>{esc(correct_behavior)}</td></tr>\n"
    else:
        html_out += "          <tr><th>Correct Behavior</th><td><span class='empty-note'>No information provided.</span></td></tr>\n"

    if critical_issues:
        html_out += f"          <tr><th>Critical Issues</th><td>{html_list(critical_issues)}</td></tr>\n"
    else:
        html_out += "          <tr><th>Critical Issues</th><td><span class='empty-note'>None noted.</span></td></tr>\n"

    if issues:
        html_out += f"          <tr><th>Other Issues (non-critical)</th><td>{html_list(issues)}</td></tr>\n"
    else:
        html_out += "          <tr><th>Other Issues (non-critical)</th><td><span class='empty-note'>None noted.</span></td></tr>\n"

    html_out += "        </tbody>\n"
    html_out += "      </table>\n"

    if not (critical_issues or issues or fac_statement or correct_behavior):
        html_out += "      <p class='empty-note'>No error assessment information recorded.</p>\n"

    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 4: File Classification
    html_out += "    <div class='section' id='sec-classification'>\n"
    html_out += "      <h2>4. File Classification</h2>\n"
    if classification_list:
        for idx, item in enumerate(classification_list, start=1):
            fp = item.get("file_path", "")
            response = item.get("response", {}) or {}
            classification = response.get("classification", "")
            description = response.get("description", "")
            rating = response.get("how_relevant_rating", "")

            html_out += f"      <h3>4.{idx} {esc(fp)}</h3>\n"
            if classification:
                html_out += f"      <p><strong>Classification:</strong> {esc(classification)}</p>\n"
            if rating != "":
                html_out += f"      <p><strong>Relevance:</strong> {esc(rating)}</p>\n"
            if description:
                html_out += f"      <p>{esc(description)}</p>\n"
    else:
        html_out += "      <p class='empty-note'>No file classification data recorded.</p>\n"
    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 5: Code & Scripts
    html_out += "    <div class='section' id='sec-code'>\n"
    html_out += "      <h2>5. Code &amp; Scripts</h2>\n"
    any_code = False
    if code:
        any_code = True
        html_out += "      <h3>5.1 Extracted Example Code (Python)</h3>\n"
        html_out += "      <pre><code class='language-python'>"
        html_out += esc(code)
        html_out += "</code></pre>\n"
    if requirements:
        any_code = True
        html_out += "      <h3>5.2 requirements.txt</h3>\n"
        html_out += "      <pre><code class='language-plaintext'>"
        html_out += esc(requirements)
        html_out += "</code></pre>\n"
    if runscript:
        any_code = True
        html_out += "      <h3>5.3 run.sh / execution script</h3>\n"
        html_out += "      <pre><code class='language-bash'>"
        html_out += esc(runscript)
        html_out += "</code></pre>\n"

    if not any_code:
        html_out += "      <p class='empty-note'>No code or scripts captured in current state.</p>\n"
    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 6: Execution Result
    html_out += "    <div class='section' id='sec-execution'>\n"
    html_out += "      <h2>6. Execution Result</h2>\n"

    # 6.1 stdout
    html_out += "      <h3>6.1 Python stdout</h3>\n"
    html_out += "      <pre><code class='language-plaintext'>"
    html_out += esc(result_python_stdout)
    html_out += "</code></pre>\n"

    # 6.2 stderr
    html_out += "      <h3>6.2 Python stderr</h3>\n"
    html_out += "      <pre><code class='language-plaintext'>"
    html_out += esc(result_python_stderr)
    html_out += "</code></pre>\n"

    # 6.3 log
    html_out += "      <h3>6.3 Execution log (JSON)</h3>\n"
    html_out += "      <pre><code class='language-json'>"
    html_out += esc_json(result_log)
    html_out += "</code></pre>\n"

    # 6.4 results_json
    html_out += "      <h3>6.4 results_json</h3>\n"
    html_out += "      <pre><code class='language-json'>"
    html_out += esc_json(result_results_json)
    html_out += "</code></pre>\n"

    # # 6.5 files downloaded (simple list)
    # html_out += "      <h3>6.5 Files downloaded</h3>\n"
    # if files_downloaded:
    #     html_out += "      <ul>\n"
    #     for rel_path, size in files_downloaded.items():
    #         full_path = os.path.join(file_path, rel_path) if file_path else rel_path
    #         esc_rel = esc(rel_path)
    #         esc_full = esc(full_path)
    #         html_out += "        <li>\n"
    #         html_out += f"          <a href='{esc_full}' target='_blank'>{esc_rel}</a>\n"
    #         # If you want to show size, uncomment:
    #         # html_out += f"          <span class='monosmall'> ({size} bytes)</span>\n"
    #         html_out += "        </li>\n"
    #     html_out += "      </ul>\n"
    # else:
    #     html_out += "      <p class='empty-note'>No files were reported as downloaded.</p>\n"

    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 7: Generated Output (from to-save directory)
    html_out += "    <div class='section' id='sec-output'>\n"
    html_out += "      <h2>7. Generated Output</h2>\n"

    if to_save_files:
        html_out += "      <p>These files were placed in the <code>to-save</code> directory.</p>\n"
        html_out += "      <ul>\n"
        for fobj in to_save_files:
            rel_name = fobj["name"]      # canonical relative name
            full_path = fobj["path"]     # base path + relative name
            is_img = fobj.get("is_image", False)

            esc_name = esc(rel_name)
            esc_full = esc(full_path)

            html_out += "        <li>\n"
            html_out += f"          <a href='{esc_full}' target='_blank'>{esc_name}</a>\n"
            if is_img:
                html_out += "          <div class='output-preview'>\n"
                html_out += f"            <img src='{esc_full}' alt='{esc_name}' class='preview-img'>\n"
                html_out += "          </div>\n"
            html_out += "        </li>\n"
        html_out += "      </ul>\n"
    else:
        html_out += "      <p class='empty-note'>No files were saved to the <code>to-save</code> directory.</p>\n"

    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 8: Cost & Runtime Summary
    html_out += "    <div class='section' id='sec-costs'>\n"
    html_out += "      <h2>8. Cost &amp; Runtime Summary</h2>\n"
    html_out += "      <table>\n"
    html_out += "        <tbody>\n"
    html_out += f"          <tr><th>Total Cost (USD)</th><td>${total_cost:0.4f}</td></tr>\n"
    html_out += f"          <tr><th>Total Runtime (seconds)</th><td>{total_runtime_secs:0.2f}</td></tr>\n"
    if costs:
        html_out += "          <tr><th>Cost Breakdown</th><td>\n"
        html_out += "            <ul>\n"
        for k, v in costs.items():
            try:
                v_float = float(v)
            except Exception:
                v_float = 0.0
            html_out += f"              <li>{esc(k)}: ${v_float:0.4f}</li>\n"
        html_out += "            </ul>\n"
        html_out += "          </td></tr>\n"
    if runtimes:
        html_out += "          <tr><th>Runtime Breakdown</th><td>\n"
        html_out += "            <ul>\n"
        # keep indent consistent
        for k, v in runtimes.items():
            try:
                v_float = float(v)
            except Exception:
                v_float = 0.0
            html_out += f"              <li>{esc(k)}: {v_float:0.2f} s</li>\n"
        html_out += "            </ul>\n"
        html_out += "          </td></tr>\n"
    html_out += "        </tbody>\n"
    html_out += "      </table>\n"
    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Section 9: Debugging Change Log (full JSON)
    html_out += "    <div class='section' id='sec-changelog'>\n"
    html_out += "      <h2>9. Debugging Change Log</h2>\n"
    if change_log:
        html_out += "      <pre><code class='language-json'>"
        html_out += esc_json(change_log)
        html_out += "</code></pre>\n"
    else:
        html_out += "      <p class='empty-note'>No change log entries recorded.</p>\n"
    html_out += "      <p class='back-to-top'><a href='#top' class='btn'>&uarr; Back to Table of Contents</a></p>\n"
    html_out += "    </div>\n"

    # Close content & body
    html_out += "  </div>\n"
    html_out += "</body>\n"
    html_out += "</html>\n"

    # Write output
    try:
        with open(filename_output_html, 'w', encoding='utf-8') as f:
            f.write(html_out)
        print(f"Wrote HTML report to {filename_output_html}")
    except Exception as e:
        print(f"Error writing HTML report to {filename_output_html}: {e}")




def make_example_from_repo_entry_point(repo_url:str, model_str_code:str="openrouter/openai/gpt-oss-120b", model_str_fast:str="openai/gpt-5-mini", max_runtime_mins:int=30, max_debug_iterations:int=6, output_base_path:str="output"):
    # Create an instance of MakeExampleFromRepo
    example_maker = MakeExampleFromRepo(repo_url, max_runtime_mins=max_runtime_mins, max_debug_iterations=max_debug_iterations, output_directory=output_base_path)

    # Begin the process
    example_maker.begin(model_str_code=model_str_code, model_str_fast=model_str_fast)



# Use a proper argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="CodeDistiller: Generate code examples from GitHub repositories.")
    parser.add_argument("repo_url", type=str, help="URL of the GitHub repository.  Example:  https://github.com/cognitiveailab/textworldexpress")
    parser.add_argument("config_file", type=str, help="Path to the configuration JSON file.  Example: config-claude45.json")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("CodeDistiller (Version: " + str(VERSION_CODEDISTILLER) + ")\n")

    # Parse command line arguments.
    args = parse_arguments()
    repo_url = args.repo_url
    config_file = args.config_file
    print(f"Making example from repository: {repo_url}")
    print(f"Using configuration file: {config_file}")

    # Load the API keys from an external file (if used)
    loadAPIKeys()

    # Load the configuration file, to retrieve the `model_str_code`, `model_str_classification`, `max_runtime_mins`, and `max_debug_iterations` parameters
    config = {}
    print("Loading configuration from: " + str(config_file))
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading {config_file}: {e}")
        exit(1)

    # Check for required keys
    required_keys = ["model_str_code", "model_str_fast", "max_runtime_mins", "max_debug_iterations", "output_base_path"]
    errors = False
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key '{key}' in make_example_config.json")
            errors = True
    if (errors == True):
        exit(1)

    # Run CodeDistiller
    make_example_from_repo_entry_point(
        repo_url=repo_url,
        model_str_code=config["model_str_code"],
        model_str_fast=config["model_str_fast"],
        max_runtime_mins=config["max_runtime_mins"],
        max_debug_iterations=config["max_debug_iterations"],
        output_base_path=config["output_base_path"]
    )

    print("Completed.")
