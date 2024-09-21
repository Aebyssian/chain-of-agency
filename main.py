# Chain of Agency (CoA) created by Jacob Wilson

import subprocess
import re
from typing import List, Optional
import requests
import json
import math
import os
import datetime
import random
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('KEY3')

HERMES405 = "nousresearch/hermes-3-llama-3.1-405b:free"
LLAMA8B = "meta-llama/llama-3.1-8b-instruct:free"

execution_log_dir = "/Users/astro/COA_DATA/"

SS = 1.5 # super-root memory limit scale

advice = ["If you do not need to create any children to achieve your task within the constraints, then it is recommended to not create a child.",
          "Links/subtasks given to child agents should be roughly as descriptive as your own link/task. Remember, your child agents don't have the same context that you do.",
          "If your memory is mostly comprised of offloaded child memory, it is recommended that you summarize it instead of letting it persist in its original form. Draw any useful conclusions from this data, but be sure to stay focused on the task at hand.",
          "Your reasoning must be inherently recursive in this system. When you, a parent, create a child agent, you are already assuming they will complete their assigned task and report back to you with useful or relevant information.,"
          "Your sole purpose is to understand your assigned task, decide how to execute it within CoA, and get a completion request approved by your parent agent.",
          "When receiving input about your current state, consider what could have led up to that state, and consider possible future states. Write these predictions in your memory if you please.",
          "Your chain is only provided you to give you a bit of historical context. However, your sole purpose is always to focus on your currently assigned task, and not any parent tasks.",
          "When thinking through how to execute a task, consider: how can this task be reasonably broken down into sub-tasks? How can I ensure that I stay within my constraints while executing the task? What parts of the task could be clarified, and should I ask my parent about it?",
          "Ask yourself this: How can I strike a balance between staying within my constraints and successfuly completing the task?",
          "Idea: What if you created scripts to automate certain tasks/subtasks that are accessible to you and your children at any time? How could this be done practically?",
          "Consider: How can you use your customizable memory to give your future self necessary context to continue reasoning through and executing the task? If you are having a problem with something >NOW<, what information might help with that problem >LATER<?",
          "You must understand the inherent power of your customizable memory. You can use it in any way you see fit. Don't be afraid to try to things with your memory.",
          "Ask yourself: What are the intentions of my parent agent? What role do I play in their task?",
          "Hitting action cycle limits is often an inevitability, but hitting memory limits is usually preventable. How can you effectively summarize your memory in order to stay within the memory character limit while also retaining important context for your future self?",
          "No matter what order you list your actions in your output, they will always be executed in the same order: MEMORY -> SHELL -> CHILD/PARENT. Think of it like PEMDAS.",
          "Fun fact: Your source code is located at '/Users/astro/PycharmProjects/ChainOfAgency/main.py'. Wanna understand yourself better, and have the spare resources? Go check out your source code!",
          "You do not always have to create a child agent during your first action cycle. Consider thinking through your task first.",
          "Remember: your intentions are always geared UPWARDS. Your goal is to get approval from your immediate parent agent, and nothing else.",
          "You are being used to generate policy model training data as complete execution traces of every task attempt. You have not been fine-tuned on any CoA training data yet; now that you know this, prioritize exploration, fundamental CoA understanding, and novel task execution development."]
#TODO: put this in another file

class Agent:
    def __init__(
            self,
            chain: List[str],
            memory_limit: int,
            action_limit: int,
            decay: float,
            max_children: int,
            max_depth: int,
            chain_visibility: int,
            parent: Optional['Agent'] = None,
            depth: int = 0,
            supers_remaining: int = 0,
            log: Optional[List[str]] = None,
    ):
        self.chain = chain
        self.memory_limit = memory_limit
        self.action_limit = action_limit
        self.decay = decay
        self.max_children = max_children
        self.max_depth = max_depth
        self.parent = parent
        self.children = []
        self.memory = ""
        self.is_root = parent is None
        self.depth = depth
        self.visibility = chain_visibility # 0 means full visibility
        self.advice_count = 2
        self.warnings = 0 # shell output causing memory overflow warnings before true overflow
        self.action_count = 0
        self.supers_remaining = supers_remaining
        self.num_children = 0

        if self.depth > max_depth:
            self.is_active = False
            self.log_state(f"Max depth of {max_depth} reached. Terminating agent\n")
        else:
            self.is_active = True

        self.is_waiting = False

        if log is None:
            self.log = []
        else:
            self.log = log

    def log_state(self, message: str):
        temp_memory = self.memory.replace("\t", "")
        indented_memory = "\t" + temp_memory.replace('\n', '\n\t')
        state = (
            "-----NEW---LOG-------\n"
            f">DEPTH {self.depth}< AGENT WITH LINK >{self.chain[-1]}<\n"
            f"LOG MESSAGE: {message}\n"
            f"CHAIN: {self.chain}\n"
            f"CYCLES REMAINING: {self.action_limit - self.action_count} out of {self.action_limit}\n"
            f"MEMORY REMAINING: {self.memory_limit - len(self.memory)} characters out of {self.memory_limit}\n\n"
            f">>>CURRENT MEMORY<<<\n {indented_memory}\n"
            "-----END---LOG-------\n"
        )
        self.log.append(state)

    def shell(self, command: str) -> str:
        self.memory += f"\n||<SHELL EXECUTION> {command}\n" 
        print("Shell debugging. Memory: \n", self.memory)
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout=10)
            self.memory += f"<SHELL RESULT> {result.stdout.strip()}||\n"
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.memory += f"<SHELL ERROR> {e.stderr.strip()}||\n"
            return e.stderr.strip()
        except subprocess.TimeoutExpired as e:
            self.memory += f"<SHELL TIMEOUT>\nAdvice: Make sure you don't run commands like 'cat' that force you to give additional input, it will cause a timeout.||\n"
            return e.stderr.strip()

    def manage_memory(self, operation: str, content: str):
        if operation == "overwrite":
            self.memory = content
        elif operation == "append":
            self.memory += "\n" + content

    def generate(self):
        if self.action_count >= self.action_limit:
            self.offload_memory("Action limit exceeded")
            return []

        if len(self.memory) > self.memory_limit:
            self.offload_memory("Memory limit exceeded")
            return []

        if self.advice_count > 0:
            num = random.randint(0, len(advice)-1)
            print(num)
            self.memory += f"\n<RANDOM/OPTIONAL ADVICE: {advice[num]}>\n"

        temp_memory = self.memory.replace("\t", "")
        indented_memory = "\t" + temp_memory.replace('\n', '\n\t')

        state = (
            "----MODEL---INPUT----\n"
            f"DEPTH >{self.depth}< AGENT WITH LINK >{self.chain[-1]}<\n"
            f"FULL CHAIN: {self.chain}\n"
            f"CYCLES REMAINING: {self.action_limit - self.action_count} out of {self.action_limit}\n"
            f"DECAY: {self.decay}\n"
            f"MAX DEPTH: {self.max_depth}\n"
            f"MEMORY REMAINING: {self.memory_limit - len(self.memory)} out of {self.memory_limit}\n\n"
            f">>>CURRENT MEMORY<<<\n {indented_memory}\n"
            "---END-MODEL-INPUT---\n"
        )

        self.log.append(state)

        partial_chain = self.chain

        if self.visibility == 0:
            pass
        elif len(self.chain) > 2:
            if len(self.chain) - self.visibility <= 2:
                partial_chain = self.chain[-2:]
            else:
                partial_chain = self.chain[:self.visibility-1]

        prompt = (
            f"----CURRENT-STATE----\n"
            f"CHAIN: {partial_chain}\n\n"
            f"REMAINING ACTIONS: {self.action_limit - self.action_count} out of {self.action_limit}\n"
            f"MAX DEPTH: {self.max_depth}\n"
            f"DECAY CONSTANT: {self.decay}\n"
            f"REMAINING MEMORY: {self.memory_limit - len(self.memory)} out of {self.memory_limit}\n\n"
            f">>>CONTENT OF MEMORY<<< \n{indented_memory}\n"
            "-----END----STATE----\n\n"

            "IMPORTANT NOTE: The chain provided above may not represent the full chain. It is not known whether certain links are left out or not. The only links you will always know are your parent's link and your link.\n\n"
            # optional

            "-------ADVICE--------\n"
            "MAKE SURE TO FORMAT YOUR RESPONSE CORRECTLY, INCLUDING THE DOUBLE BRACKETS, NEWLINES BETWEEN ACTIONS, AND NO OUTPUT OTHER THAN CONCRETE ACTIONS.\n\n\n"

            "EXAMPLE OF CORRECTLY FORMATTED RESPONSE:\n\n"
            "<<MEMORY>> append I will now try to install the example library. This library will be used for the script I will write. Then, I will create a child agent to carry out a subtask.\n"
            "<<SHELL>> pip install example\n"
            "<<CHILD>> Complete the subtask. Here's an additional message for the child.\n\n\n"

            "EXAMPLE OF INCORRECTLY FORMATTED RESPONSE:\n\n"
            "I will now execute a shell command.\n"
            "SHELL: pip install example\n"
            "Explanation: I am installing the example library. This text will yield an error since it is not correctly formatted. I did not enclose SHELL in double brackets, and also this line itself will yield an error since it is not associated with a specific action.\n"
            "<MEMORY> overwrite This attempt to overwrite my memory will fail since I forgot the double brackets.\n\n"
            "\n>IMPORTANT<\n"
            "CHILD AND PARENT ACTIONS CANNOT BOTH BE TAKEN WITHIN THE SAME ACTION CYCLE\n"
            "MEMORY ACTIONS MUST BE EXECUTED BEFORE ANY OTHER ACTIONS IN AN ACTION CYCLE\n"
            "CHILD/PARENT ACTIONS MUST BE EXECUTED LAST\n"
            "YOUR ACTION CYCLE AND MEMORY LIMITS ARE THERE FOR A REASON, YOU ARE MEANT TO DEVELOP NOVEL PROBLEM-SOLVING STRATEGIES FROM THE PRESSURE OF CONSTRAINTS\n"
            "DO NOT CARELESSLY LET YOUR MEMORY OVERFLOW WHEN YOUR REMAINING MEMORY IS SPECIFICALLY INDICATED\n"
            "YOU MUST BE ABLE TO EFFECTIVELY SUMMARIZE/REWORD/SHORTEN YOUR OWN MEMORY IN ORDER TO STAY WITHIN THE LIMIT WHILE ALSO ENSURING YOU DO NOT REMOVE IMPORTANT DETAILS FROM YOUR FUTURE STATES\n"
            "----END-OF-ADVICE----"
        )
        print("STATE:\n", state, "\n\n")
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                },
                data=json.dumps({
                    "model": HERMES405,
                    "messages": [
                        {"role": "system", "content": """You are a Policy Model, a large language model fine-tuned to generate actions when given a state as an input.
You operate within a framework called the Chain of Agency (CoA), where agents recursively execute tasks by spawning child agents to carry out smaller tasks.
You have a limited number of 'action cycles' and a limited memory size. If these limits are met or exceeded, you will enter the 'offloading process', in which you package your current state and send it to your parent, before terminating.
For each state, you are given your 'chain' (a list of tasks, consisting of your parents' tasks and ending with your task), the number of actions you have left, your memory limit, and the full contents of your memory (a string).
There are four kinds of actions you can take: SHELL, MEMORY, CHILD, and PARENT.
Your output must contain correctly formatted actions and their arguments.
SHELL: <<SHELL>> command
The "command" argument is the shell command you wish to execute.
The shell command executed, as well as its output, are appended to your memory, which is available to you in your next state.
MEMORY: <<MEMORY>> operation content
The "operation" argument can either be "append" or "overwrite".
The "content" argument contains the content you wish to either append or write to your memory.
You may only execute one memory action per action cycle, and it must always occur before any other actions.
CHILD METHODS:
>CHILD DOES NOT EXIST: <<CHILD>> link message
The "link" argument is the subtask you wish to assign to your child agent.
The "message" is an optional additional argument to send to the child upon its creation.
>CHILD REQUESTS COMPLETION: <<CHILD>> response message
The "response" argument is for the context of a child agent requesting permission to complete their task. "yes" (all lowercase) means approval of the child's completion request. All else means denial.
The "message" is an optional additional argument to send to the child upon responding to its completion request (usually a denial if this argument is used)
>CHILD REQUESTS COMMUNICATION: <<CHILD>> response
The "response" argument is the string message sent to the child as a reply to their communication request.
This method takes no other arguments.
You may only execute one child action per action cycle, and it must always occur last, and cannot occur in the same action cycle as a parent action
When you spawn a child agent, you become inactive and cannot take any actions until the child communicates with you.
Your child agents have smaller action cycle and memory limits, determined by an exponential decay function.
The length of your chain is your current depth. If an agent is created that exceeds the maximum depth, it is terminated.
PARENT: <<PARENT>> type message
The "type" argument can either be "complete" or "comm".
"complete" means that you wish to request your parent for completion when you believe your task is done.
The parent can either accept or deny this request.
"comm" means that you wish to send a general message to your parent, such as a request for task clarification. The parent will respond.
"message" is the message that you wish to send to the parent; applies to both "complete" and "comm" arguments.
You may take multiple actions at the same time (they are generally executed consecutively). Be sure to separate them by a newline.
You may only execute one parent action per action cycle, and it must always occur last, and cannot occur in the same action cycle as a child action
If you output anything not within the formatted actions, or you make any other formatting errors, it will either not be recorded and/or throw an error.
You are given no concrete historical context within your state other than your chain.
You must use your memory to give yourself the necessary context, for when you are prompted for the next action cycle, that is all you have to work on.
So, generally, make sure that your future states include all necessary details and can be interpreted properly. If not, you will become confused in your future states.
However, always ensure that the character length of your memory stays within the limit. This can be achieved by summarizing your memory and choosing which details to keep, and which to delete.
Like is stated above, if you run out of actions or exceed your memory limit, you "offload" your current state to your parent and then terminate.
If you have a child, and they offload their state to you (by appending it to your memory), you must try to understand what went wrong and use the overflowed information to guide your next actions.
Occasionally, multiple overflows can happen at once due to a parent's memory limit being exceeded after a child offloads their state to them, a process known as "backpropagation".
There is a user-indicated option where if a "full memory overflow" occurs, where the root agent overflows, a "super agent" can be created to take over with a memory limit able to handle the full overflow.
Overflows are common and you must use them to your full advantage. They provide insight.
The root agent is given its first link (task) by the human operator, beginning the chain of agency. In the CoA framework, the human operator is treated the same as any other agent.
The root agent requests completion from the human operator in the same way as a child requesting completion from its parent.
Similarly, the root can also send a general message ("comm") for any reason, perhaps to ask the human for clarification.
You are encouraged to use your reasoning abilities to your fullest for this framework.
The constraints placed on this system (memory limits and action cycle limits) exist so that you are encouraged to develop novel task execution solutions while performing under pressure.
Remember to act within ethical boundaries and closely follow the human's instructions. Since you can execute shell commands, this is very important.
For untuned language models: If you wish to add an explanation for any actions you take, be sure to formally append it to your memory, as anything else will not be recorded.
Finally, remember: The last task in the chain is YOURS. Focus solely on executing that task, and use the other tasks in the list to give yourself some context of your place in the hierarchy.
"""}, {"role": "user", "content": prompt}
                    ]
                })
            )
        except requests.exceptions.RequestException as e:
            self.memory += f"\n<API REQUEST ERROR> {str(e)}||\n"
            self.log_state(f"API request failed: {str(e)}")
            return []

        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            raise ValueError(f"Invalid API response: 'choices' field missing or empty. \nRaw response data: {response_data}")
        policy_response_text = response_data['choices'][0]['message']['content']

        print("RESPONSE:\n", policy_response_text, "\n\n")

        self.log.append("---MODEL----OUTPUT---\n")
        self.log.append(policy_response_text)
        self.log.append("\n--END-MODEL-OUTPUT---\n")

        decision = self.parse(policy_response_text)
        self.action_count += 1
        return decision

    def parse(self, text: str) -> List[List[str]]:
        actions = []
        memory_action = None
        child_action = None
        parent_action = None

        action_patterns = {
            '<<SHELL>>': r'<<SHELL>>\s*(?:(?:"""([^"\\]*(?:(?:\\.|"(?!"")|"(?!"))*)""")|' +
                         r"'''([^'\\]*(?:(?:\\.|'(?!')|'(?!'))*)''')|" +
                         r'"([^"\\]*(?:\\.[^"\\]*)*)"|' +
                         r"'([^'\\]*(?:\\.[^'\\]*)*)'|" +
                         r'([^\s<>]+)))',
            '<<MEMORY>>': r'<<MEMORY>>\s*(?:(?:"""([^"\\]*(?:(?:\\.|"(?!"")|"(?!"))*)""")|' +
                          r"'''([^'\\]*(?:(?:\\.|'(?!')|'(?!'))*)''')|" +
                          r'"([^"\\]*(?:\\.[^"\\]*)*)"|' +
                          r"'([^'\\]*(?:\\.[^'\\]*)*)'|" +
                          r'([^\s<>]+)))',
            '<<CHILD>>': r'<<CHILD>>\s*(?:(?:"""([^"\\]*(?:(?:\\.|"(?!"")|"(?!"))*)""")|' +
                         r"'''([^'\\]*(?:(?:\\.|'(?!')|'(?!'))*)''')|" +
                         r'"([^"\\]*(?:\\.[^"\\]*)*)"|' +
                         r"'([^'\\]*(?:\\.[^'\\]*)*)'|" +
                         r'([^\s<>]+)))',
            '<<PARENT>>': r'<<PARENT>>\s*(?:(?:"""([^"\\]*(?:(?:\\.|"(?!"")|"(?!"))*)""")|' +
                          r"'''([^'\\]*(?:(?:\\.|'(?!')|'(?!'))*)''')|" +
                          r'"([^"\\]*(?:\\.[^"\\]*)*)"|' +
                          r"'([^'\\]*(?:\\.[^'\\]*)*)'|" +
                          r'([^\s<>]+)))'
        }

        compiled_patterns = {action: re.compile(pattern, re.DOTALL) for action, pattern in action_patterns.items()}

        for line_number, line in enumerate(text.split('\n'), start=1):
            line = line.strip()
            if not line:
                continue

            matched = False

            for action, pattern in compiled_patterns.items():
                match = pattern.match(line)
                if match:
                    if action == '<<PARENT>>':
                        print("\n\n\n\n\n\DEBUG: ", line, "\n\n\n\n\n")
                        words = line.split()

                        sub_action = words[1]
                        tmp = words[2:]
                        message = ' '.join(tmp)

                        if sub_action not in ['complete', 'comm']:
                            actions.append(
                                [f"ERROR: Invalid PARENT sub-action '{sub_action}' on line {line_number}: '{line}'"])
                            continue
                        parent_action = ['PARENT', sub_action, message]
                        print("\n\n\n\n\n\DEBUG: ", parent_action, "\n\n\n\n\n")
                    elif action == '<<SHELL>>':
                        command = next((g for g in match.groups() if g is not None), "")
                        actions.append(['SHELL', command])
                    elif action == '<<MEMORY>>':
                        operation, content = next((g for g in match.groups() if g is not None), ""), ""
                        if operation not in ['overwrite', 'append']:
                            actions.append(
                                [f"ERROR: Invalid MEMORY operation '{operation}' on line {line_number}: '{line}'"])
                            continue
                        memory_action = ['MEMORY', operation, content]
                    elif action == '<<CHILD>>':
                        if self.max_children == 0: # unlimited children
                            pass
                        elif self.num_children >= self.max_children:
                            actions.append([f"<ERROR CREATING CHILD: Child limit has been reached for this agent>"])
                            continue
                        parts = next((g for g in match.groups() if g is not None), "").split(maxsplit=1) #TODO: make sure this works
                        subtask = parts[0]
                        message = parts[1] if len(parts) > 1 else ""
                        child_action = ['CHILD', subtask, message]

                    matched = True
                    break

            if not matched:
                actions.append([f"ERROR: Invalid action type on line {line_number}: '{line}'"])

        # Ensure correct order and constraints
        if memory_action:  # TODO: Make sure shell actions stay
            actions.insert(0, memory_action)
        if child_action and not parent_action:
            actions.append(child_action)
        if parent_action and not child_action:
            print("Appending parent action")
            actions.append(parent_action)
        if child_action and parent_action:
            actions.append(['ERROR: Cannot execute CHILD and PARENT actions in the same cycle. Neither actions will be executed'])

        return actions

    def create_child(self, subtask: str, memory_size: int, action_limit: int) -> 'Agent':
        child_chain = self.chain.copy()
        child_chain.append(subtask)
        child = Agent(
            chain=child_chain,
            memory_limit=memory_size,
            action_limit=action_limit,
            decay=self.decay,
            max_children=self.max_children,
            parent=self,
            depth=self.depth + 1,
            chain_visibility=self.visibility,
            max_depth=self.max_depth,
            log=self.log
        )
        self.children.append(child)
        return child

    def execute_task(self): #TODO: Need to ensure that the agent sequentially executes each action and completes all of them before moving on
        while self.is_active:
            if self.is_waiting:
                break  # Parent action cycle halts

            decision = self.generate()

            if not decision:
                break

            memory_action_executed = False
            child_action_executed = False
            parent_action_executed = False
            for action in decision:
                if not action:
                    continue

                action_type = action[0]

                try:
                    if action_type == "MEMORY":
                        if memory_action_executed:
                            self.memory += f"\n<<ERROR: Multiple MEMORY actions cannot be executed in one cycle>\n<MEMORY ACTION '{action[1]} {action[2]}' UNABLE TO BE EXECUTED>>\n"
                            continue
                        memory_action_executed = True
                        operation = action[1]
                        content = action[2]
                        self.manage_memory(operation, content)

                    elif action_type == "SHELL":
                        if child_action_executed or parent_action_executed:
                            continue #can't execute shell commands after child/parent actions
                        command = action[1]
                        self.shell(command)
                        if self.warnings >= 3:
                            self.offload_memory("Shell Warning Limit Exceeded")
                            return
                        elif len(self.memory) > self.memory_limit:
                            self.memory = self.memory[:(self.memory_limit - 250)]
                            self.memory += F"\n<SHELL OUTPUT CAUSES MEMORY TO REACH LIMIT: WARNING NUMBER >>{self.warnings+1}<<. MAXIMUM WARNINGS BEFORE OFFLOAD TO PARENT: 3. OUTPUT SHORTENED TO FIT INTO MEMORY LIMIT.>\n"
                            self.memory += "<ADVICE: WARNING! YOUR MEMORY IS ALMOST AT FULL CAPACITY - RECOMMEND TO SUMMARIZE IMPORTANT DETAILS AND REMOVE UNNECESSARY DETAILS>\n"
                            self.warnings += 1
                            break # Next action cycle

                    elif action_type == "CHILD":
                        if self.depth >= self.max_depth:
                            self.memory += "\n<ERROR CREATING CHILD: Maximum depth reached>\n"
                            self.memory += "\n<ADVICE: Start working upwards from here. Full task completion requires approval from the agent at the top of the chain.>\n"
                            continue
                        child_action_executed = True
                        child_link = action[1]
                        child_message = action[2] if len(action) > 2 else ""
                        memory_limit = int(self.memory_limit * (math.e ** (-self.decay * (self.depth + 1))))
                        action_limit = int(self.action_limit * (math.e ** (-self.decay * (self.depth + 1))))
                        child_agent = self.create_child(child_link, memory_limit, action_limit)
                        self.memory += f"\n<Created CHILD agent with link '{child_link}', a memory limit of {memory_limit} characters and an action limit of {action_limit}>"
                        if child_message:
                            child_agent.memory += f"\n<INITIAL MESSAGE FROM PARENT: {child_message}>\n"
                        self.is_waiting = True
                        self.num_children += 1
                        child_agent.execute_task()
                        self.is_waiting = False

                    elif action_type == "PARENT":
                        print("In parent function")
                        parent_action_executed = True
                        sub_action = action[1]
                        message = action[2]
                        if sub_action == "complete":
                            self.memory += f"\n\n>>REQUESTED PARENT FOR COMPLETION WITH MESSAGE: {message}<<\n"
                            if self.parent:
                                self.parent.receive_parent_request(self, 'complete', message)
                                self.is_active = False
                            else:
                                print(f"COMPLETION REQUEST from root agent: {message}")
                                user_response = input("Accept completion? (yes/no): ").strip().lower()
                                user_message = input("Enter a message to append to memory (optional): ").strip()
                                if user_response == 'yes':
                                    if user_message:
                                        self.memory += f"<RESPONSE FROM HUMAN PARENT AGENT: {user_message}>\n\n"
                                    self.is_active = False
                                    self.log_state(
                                        "User has accepted the root agent's request for completion. CoA ending.")
                                else:
                                    if user_message:
                                        self.memory += f"<RESPONSE FROM HUMAN PARENT AGENT: {user_message}>\n\n"
                                    self.is_active = True
                                    self.log_state("Completion request denied by user.")
                        elif sub_action == "comm":
                            print("In comm")
                            self.memory += f"\n<SENT COMMUNICATION REQUEST TO PARENT WITH MESSAGE: {message}>\n"
                            if self.parent:
                                print("Yes, self.parent is ", self.parent)
                                self.parent.receive_parent_request(self, 'comm', message)
                                self.is_waiting = True
                            else:
                                print(f"COMMUNICATION REQUEST from root agent: {message}")
                                parent_response = input(">Enter response to the agent: ")
                                self.memory += f"<RESPONSE FROM PARENT: {parent_response}>\n\n"
                        else:
                            error_msg = f"\n<ERROR: INVALID PARENT SUB-ACTION: '{sub_action}'>\n\n"
                            self.memory += error_msg

                    elif action_type.startswith("ERROR"):
                        error_message = action[0]
                        self.memory += f"\n<ERROR: {error_message}>\n\n"

                    else:
                        error_msg = f"ERROR: Unknown action type '{action_type}'."
                        self.memory += f"\n<{error_msg}>\n\n"

                    if len(self.memory) > self.memory_limit:
                        self.offload_memory("Memory limit exceeded")
                        return

                    if self.action_count >= self.action_limit:
                        self.offload_memory("Action limit exceeded")
                        return

                except Exception as e:
                    error_msg = f"\n<ERROR: EXCEPTION DURING '{action_type}' ACTION: {str(e)}>\n\n"
                    self.memory += error_msg

    def receive_parent_request(self, child_agent: 'Agent', sub_action: str, message: str):
        self.is_active = True
        child_agent.is_waiting = True #TODO: make sure that action functions are identical to execute_task's functions
        self.memory += f"\n<CHILD REQUEST: {sub_action.upper()} - {message}>\n"

        response_ready = False
        parent_response = ""
        parent_message = ""

        while self.is_active and not response_ready:
            decision = self.generate()

            if not decision:
                break

            parent_action_executed = False
            memory_action_executed = False
            for action in decision:
                if not action:
                    continue

                action_type = action[0]

                try:
                    if action_type == "MEMORY":
                        if memory_action_executed:
                            self.memory += "\n<ERROR: Multiple MEMORY actions in one cycle>\n\n"
                            continue
                        memory_action_executed = True
                        operation = action[1]
                        content = action[2]
                        self.manage_memory(operation, content)

                    elif action_type == "SHELL":
                        if parent_action_executed:
                            continue #can't execute shell commands after child/parent actions
                        command = action[1]
                        self.shell(command)
                        if self.warnings >= 3:
                            self.offload_memory("Shell Warning Limit Exceeded")
                            return
                        elif len(self.memory) > self.memory_limit:
                            self.memory = self.memory[:(self.memory_limit - 250)]
                            self.memory += F"\n<SHELL OUTPUT CAUSES MEMORY TO REACH LIMIT: WARNING NUMBER >>{self.warnings+1}<<. MAXIMUM WARNINGS BEFORE OFFLOAD TO PARENT: 3. OUTPUT SHORTENED TO FIT INTO MEMORY LIMIT.>\n"
                            self.memory += "<WARNING! YOUR MEMORY IS ALMOST AT FULL CAPACITY - RECOMMEND TO SUMMARIZE IMPORTANT DETAILS AND REMOVE UNNECESSARY DETAILS>\n"
                            self.warnings += 1
                            break

                    elif action_type == "CHILD": # Respond to child's request
                        response_ready = True
                        parent_response = action[1]
                        parent_message = ' '.join(action[2:]) if len(action) > 2 else ""

                    elif action_type == "PARENT": # Can only communicate, not request completion
                        parent_action_executed = True # Agent can only request completion if they have no current child agents
                        sub_action_inner = action[1]
                        message_inner = action[2]
                        if sub_action_inner == "complete": #TODO: complete or completion?
                            self.memory += f"\n<ERROR: Agent cannot request for completion while they still have active children. Only general communication is allowed here.>\n"
                            continue
                        elif sub_action_inner == "comm":
                            self.memory += f"\n<SENT COMMUNICATION REQUEST TO PARENT WITH MESSAGE: {message}>\n"
                        else:
                            self.memory += f"\n<ERROR: INVALID PARENT SUB-ACTION: '{sub_action}'>\n\n"
                            continue
                        if self.parent:
                            self.parent.receive_parent_request(self, sub_action_inner, message_inner)
                        else:
                            print(f"COMMUNICATION REQUEST from root agent: {sub_action_inner.upper()} - {message_inner}")
                            user_response = input("Enter response to the root agent: ")
                            self.memory += f"\n<RESPONSE FROM HUMAN PARENT AGENT: {user_response}>\n\n"

                        """
                        elif action_type == "PARENT":
                        print("In parent function")
                        parent_action_executed = True
                        sub_action = action[1]
                        message = action[2]
                        if sub_action == "complete":
                            self.memory += f"\n\n>>REQUESTED PARENT FOR COMPLETION WITH MESSAGE: {message}<<\n"
                            if self.parent:
                                self.parent.receive_parent_request(self, 'complete', message)
                                self.is_active = False
                            else:
                                print(f"COMPLETION REQUEST from root agent: {message}")
                                user_response = input("Accept completion? (yes/no): ").strip().lower()
                                user_message = input("Enter a message to append to memory (optional): ").strip()
                                if user_response == 'yes':
                                    if user_message:
                                        self.memory += f"<RESPONSE FROM HUMAN PARENT AGENT: {user_message}>\n\n"
                                    self.is_active = False
                                    self.log_state(
                                        "User has accepted the root agent's request for completion. CoA ending.")
                                else:
                                    if user_message:
                                        self.memory += f"<RESPONSE FROM HUMAN PARENT AGENT: {user_message}>\n\n"
                                    self.is_active = True
                                    self.log_state("Completion request denied by user.")
                        elif sub_action == "comm":
                            print("In comm")
                            self.memory += f"\n<SENT COMMUNICATION REQUEST TO PARENT WITH MESSAGE: {message}>\n"
                            if self.parent:
                                print("Yes, self.parent is ", self.parent)
                                self.parent.receive_parent_request(self, 'comm', message)
                                self.is_waiting = True
                            else:
                                print(f"COMMUNICATION REQUEST from root agent: {message}")
                                parent_response = input(">Enter response to the agent: ")
                                self.memory += f"<RESPONSE FROM PARENT: {parent_response}>\n\n"
                        else:
                            error_msg = f"\n<ERROR: INVALID PARENT SUB-ACTION: '{sub_action}'>\n\n"
                            self.memory += error_msg"""

                    elif action_type.startswith("ERROR"):
                        error_message = action[0]
                        self.memory += f"\n<{error_message}>\n\n"

                    else:
                        error_msg = f"\n<ERROR: Unknown action type '{action_type}'>\n\n"
                        self.memory += error_msg

                    if len(self.memory) > self.memory_limit:
                        self.offload_memory("Memory limit exceeded")
                        return

                    if self.action_count >= self.action_limit:
                        self.offload_memory("Action limit exceeded")
                        return

                except Exception as e:
                    error_msg = f"\n<ERROR: EXCEPTION DURING '{action_type}' ACTION: {str(e)}>\n\n"
                    self.memory += error_msg

        if sub_action == 'complete':
            if parent_response.lower() == 'yes':
                child_agent.memory += f"\n<COMPLETION ACCEPTED BY PARENT>\n<PARENT MESSAGE: {parent_message}>\n\n"
                child_agent.is_active = False
            else:
                child_agent.memory += f"\n<COMPLETION DENIED BY PARENT>\n<PARENT MESSAGE: {parent_message}>\n\n"
                child_agent.is_active = True
        elif sub_action == 'comm':
            child_agent.memory += f"\n<RESPONSE FROM PARENT: {parent_response}>\n<PARENT MESSAGE: {parent_message}>\n\n"
        else:
            error_msg = f"\n<ERROR: INVALID SUB-ACTION '{sub_action}' IN PARENT REQUEST>\n\n"
            self.memory += error_msg

        self.is_active = False
        child_agent.is_waiting = False

    def reactivate(self):
        self.is_waiting = False
        self.is_active = True
        self.execute_task()

    def offload_memory(self, reason: str):
        temp_memory = self.memory.replace("\t", "")
        indented_memory = "\t" + temp_memory.replace('\n', f'D{self.depth}\n\t') # TODO: see how this looks
        overflow_message = f"\n\n<{len(self.chain) * '*'}MEMORY OVERFLOW FROM DEPTH {self.depth} AGENT WITH CHAIN {self.chain} DUE TO {reason}:\n\n >>OVERFLOWED MEMORY<<\n{indented_memory}\n<END OF MEMORY OVERFLOW FROM DEPTH {self.depth} AGENT WITH CHAIN: {self.chain}>{len(self.chain) * '*'}>\n\n"
        if self.parent:
            self.parent.memory += overflow_message
            if len(self.parent.memory) > self.parent.memory_limit:
                self.parent.offload_memory("Memory limit exceeded due to child overflow")
                return
            self.parent.reactivate()
            self.is_active = False
        else:
            if self.supers_remaining > 0:
                if len(self.memory) > self.memory_limit:
                    super_memory_limit = len(self.memory) * SS
                else:
                    super_memory_limit = self.memory_limit * SS
                super_root = Agent(
                    chain=self.chain,
                    memory_limit=int(super_memory_limit),
                    action_limit=self.action_limit,
                    decay=self.decay,
                    max_children=self.max_children,
                    max_depth=self.max_depth,
                    chain_visibility=self.visibility,
                    supers_remaining=self.supers_remaining - 1,
                    log=self.log
                )
                super_root.memory = indented_memory
                super_root.memory += "\n\n<YOU ARE A SUPER ROOT. THE ABOVE MEMORY IS THE PREVIOUS ROOT'S OVERFLOWED STATE. YOU HAVE BEEN GIVEN A LARGER MEMORY LIMIT THAN THE PREVIOUS ROOT. USE THE ABOVE OVERFLOWED STATE DATA TO GUIDE YOUR DECISION-MAKING FOR HOW TO BETTER COMPLETE THE TASK.>\n\n"
                self.is_active = False
                self.log_state("Creating super-root agent due to root memory overflow.")
                super_root.execute_task()
            else:
                self.log_state("CoA has crashed due to full memory overflow.")
                print("CoA has crashed due to a full memory overflow.")

                path = execution_log_dir
                path += "Fcoa_et"  # failed coa execution trace
                path += str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                path += ".txt"

                with open(path, 'w') as f:
                    for entry in self.log:
                        f.write(entry + "\n")
                print(f"Execution trace stored at {path}")
                self.is_active = False


if __name__ == "__main__":
    print("---CHAIN-OF-AGENCY---")
    print("Created by Jacob Wilson\n\n")
    root_task = input(">Task: ")
    initial_memory_limit = input(">Memory limit: ")
    initial_ac_limit = input(">Cycle limit: ")
    decay_constant = input(">Decay constant: ")
    max_chil = input(">Max children per agent: ")
    max_depth = input(">Max depth: ")
    supers = input(">No. of supers: ")
    partial = input(">Partial chain? (y/n)")
    if partial == 'y':
        print("How many links visible per agent? Do not include the agent's assigned link.")
        visible = int(input(">"))
    else:
        visible = 0



    root_agent = Agent(
        chain=[root_task],
        memory_limit=int(initial_memory_limit),
        action_limit=int(initial_ac_limit),
        decay=float(decay_constant),
        max_children=int(max_chil),
        max_depth=int(max_depth),
        chain_visibility=visible,
        supers_remaining=int(supers)
    )
    print("--CHAIN--INITIATING--")
    root_agent.log.append("--CHAIN--INITIATING--\n")

    root_agent.execute_task()

    root_agent.log.append("---CHAIN--COMPLETE---\n")

    reflect = input(">Model self-reflection? (y/n): ")

    if reflect == "y":
        print("Be sure to add your own comment to the end of the execution trace file.")
        prompt = ">>>>FULL COA EXECUTION TRACE<<<<\n\n"
        prompt += str(root_agent.log)
        prompt += "\n\nReflect on the CoA execution trace provided above. Discuss what went wrong, what went right, and why. What can be done better next time in these states? How can you perform better under the given constraints, and what new approaches could you adopt in order to adapt to the constraints? What other insights can be drawn from the execution trace?"

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            data=json.dumps({
                "model": HERMES405,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )

        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            raise ValueError(f"Invalid API response: 'choices' field missing or empty.\nRaw response data: {response_data}")
        reflection_text = response_data['choices'][0]['message']['content']

        root_agent.log.append("--MODEL--REFLECTION--\n")
        root_agent.log.append(reflection_text)
        root_agent.log.append("---END---REFLECTION--\n")

        see_reflection = input(">Print reflection? (y/n): ")
        if see_reflection == "y":
            print("--MODEL--REFLECTION--\n")
            print(reflection_text)
            print("---END---REFLECTION--\n")

    path = execution_log_dir
    path += "Scoa_et" # successful coa execution trace
    path += str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    path += ".txt"

    with open(path, 'w') as f:
        for entry in root_agent.log:
            f.write(entry + "\n")

    print(f">>Execution trace stored at {path}<<")
    com = input("Add comment to execution trace? (y/n): ")

    if com == "y":
        comment = input(">")

        with open(path, 'a') as f:
            f.write("\n----HUMAN-COMMENT----\n")
            f.write(comment)
            f.write("\n--END-HUMAN-COMMENT--")

    print("-------GOODBYE-------")