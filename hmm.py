ddir = "/bt3102project/submission"

#%%
import json
import pandas as pd
import numpy as np
import random
import math

class HMM:
    def __init__(self, possible_states, possible_emissions,seed):
        """
        Possible tags is a list of possible tags for the hidden states in this HMM
        Possible emissions is a list of all possible emissions
        """
        # We initailise by seeding the random function with the provided seed
        random.seed(seed)

        def generate_n_probabilities(n):
            """
            Helper function. Generates a list of n probabilities, the total of which sums to 1
            """
            r = []
            for i in range(n):
                new_random_number = random.random()
                r.append(new_random_number)
                random.seed(new_random_number) # Seed the next number with the current random number
            s = sum(r)
            r = [ i/s for i in r ]
            return r

        # Handle unseen output
        possible_emissions.append("NONE")

        # Generate initial random probabilities for transition, emission, and pi
        transition_probs_init = { state: generate_n_probabilities(len(possible_states + ["STOP"])) for state in possible_states}
        emission_probs_init = {state: generate_n_probabilities(len(possible_emissions)) for state in possible_states}
        pi_init = generate_n_probabilities(len(possible_states))

        # Initialise transition probs, emission probs, and the probability of transiting from START to state_i, pi
        self.transition_probs = { state_i: { 
            state_j : transition_probs_init[state_i][index]
            for (index,state_j) in enumerate(possible_states + ["STOP"]) } 
            for state_i in possible_states }
        self.emission_probs = { state : { 
            emission: emission_probs_init[state][index] 
            for (index,emission) in enumerate(possible_emissions)} 
            for state in possible_states }
        self.pi = {state: pi_init[index] for (index,state) in enumerate(possible_states)}

        # Store the possible states and possible emissions for easy access in class methods
        self.possible_states = possible_states
        self.possible_emissions = possible_emissions

    def forward_backward_algorithm(self,sequences,max_iter,thresh):
        """
        Runs the forward backward algorithm (though all sequences in the sequences variable, which is a list of lists)
        This method terminates when max_iter is reached, or when the incremental improvement to the probability of seeing the data drops below thresh
        """

        def update_probs(accumulated_start_zai, accumulated_zai, accumulated_gamma):
            # Update transition probabilities
            for state_i in self.possible_states:
                for state_j in self.possible_states + ["STOP"]:
                    self.transition_probs[state_i][state_j] = accumulated_zai[state_i][state_j]/ sum(accumulated_zai[state_i].values())

            # Update probability for transiting from START to state_i
            for state_i in self.possible_states:
                self.pi[state_i] = accumulated_start_zai[state_i]/sum(accumulated_start_zai.values())

            # Update emission probabilities
            for state_j in self.possible_states:
                for emission in self.possible_emissions:
                    self.emission_probs[state_j][emission] = accumulated_gamma[state_j][emission]/sum(accumulated_gamma[state_j].values())
                
                # Set emission prob for NONE to minimum
                min_prob = min(accumulated_gamma[state_j].values())
                self.emission_probs[state_j]["NONE"] = min_prob/sum(accumulated_gamma[state_j].values())

        # Initialise the first item in the log probability list to negative infinity; Log probabilities will be appended here after each iteration
        # Algorithm will terminate if the difference between latest log probability and the previous log probability is less than thresh
        iter_log_probs = [-math.inf]
        
        for i in range(max_iter):

            # Initialise iteration log probability to 0. Also initialise accumulated_start_zai, accumulated_zai, and accumulated_gamma
            # Accumulated_start_zai is a corner case of zai. It is used for the updating of P ( y1 = j ) - ie the initial probabilities, pi.
            iter_log_prob = 0
            accumulated_start_zai = { state_i : 0 for state_i in self.possible_states }
            accumulated_zai = { state_i : {state_j : 0 for state_j in self.possible_states + ["STOP"]} for state_i in self.possible_states }
            accumulated_gamma = { state: { emission : 0 for emission in self.possible_emissions } for state in self.possible_states }

            # Go through all the sequences in our data. The forward backward step function returns the log probability for seeing the sequence that was passed to it
            # We add this log probability to the iteration log probability, since the log probability of all the sequences in the sum of the log probability of each sequence
            for sequence in sequences:
                iter_log_prob += self.forward_backward_step(sequence, accumulated_start_zai, accumulated_zai, accumulated_gamma)
            
            # After each sequence, we use the zai and gamma values to update the transition and output probabilities. This concludes the iteration
            update_probs(accumulated_start_zai,accumulated_zai,accumulated_gamma)
            
            # Print the log probability for this iteration. Can be removed when not needed anymore
            print("Iteration {} : {}".format(i,iter_log_prob))

            # Terminate the algorithm if at convergence (improvement in log probability less than threshold)
            if iter_log_prob - iter_log_probs[-1] < thresh:
                break
            iter_log_probs.append(iter_log_prob)

    def forward_backward_step(self, sequence, accumulated_start_zai, accumulated_zai, accumulated_gamma):
        """
        Runs forward backward for one sequence, returning zai and gamma. 
        In most cases, this method should not be called directly. Instead, refer to the forward_back_algorithm method
        """

        def forward_probabilities(sequence):
            """
            Takes in a sequence and returns a dictionary, forward[t][j],
            which represents alpha[t][j] (ie. the probability that state at time step t is j, given the sequence of observations up till time step t)
            """
            states = self.transition_probs.keys()
            forward = {index: { state : None for state in states } for index in range(1, len(sequence) + 1 )}
            
            # Initialisation
            first_observation = sequence[0]
            for state in states:
                forward[1][state] = self.pi[state] * self.emission_probs[state][first_observation]

            # Compute forward prob at each time step
            for time_step in range(2,len(sequence) + 1):
                current_observation = sequence[time_step-1]
                for state in states:
                    forward[time_step][state] = sum(
                        [ forward[time_step - 1][prev_state] * 
                        self.transition_probs[prev_state][state] * 
                        self.emission_probs[state][current_observation] for prev_state in states ]
                    )

            return forward


        def backward_probabilities(sequence):
            """
            Takes in a sequence and returns a dictionary, backward[t][i],
            which represents beta[t][i] (ie. the probability that the state at time step t is i, given the sequence of observations from t+1 to n)
            """

            states = self.transition_probs.keys()
            backward = {index: { state : None for state in states } for index in range(1, len(sequence) + 1 )}

            # Initialisation
            for state in states:
                nth_time_step = len(sequence)
                backward[nth_time_step][state] = self.transition_probs[state]["STOP"]

            # Compute backward prob at each time step
            for time_step in range(len(sequence) - 1, 0, -1):
                next_observation = sequence[time_step]
                for state in states:
                    backward[time_step][state] = sum(
                        [ backward[time_step + 1][next_state] * 
                        self.transition_probs[state][next_state] * 
                        self.emission_probs[next_state][next_observation] for next_state in states ]
                    )

            return backward
            

        def compute_zai(sequence, alpha, beta):
            """
            Takes in a sequence and returns a dictionary, sum_zai[i][j],
            which represents the summed probability of transiting from i to j across all time steps
            """
            states = list(self.transition_probs.keys())

            # Instead of storing zai values for each time step, we use the sum_zai dictionary to store transitions from state i to state j across all time steps, 
            # since that is what is ultimately required in computation. By doing so, we save the need to iterate through all time steps and sum it up later on,
            # which would be computationally expensive
            sum_zai = { state_i : {state_j : 0 for state_j in states + ["STOP"]} for state_i in states } 

            joint_prob = sum([ alpha[len(sequence)][state] * self.transition_probs[state]["STOP"] for state in self.transition_probs])

            for time_step in range(1, len(sequence)):
                emission_at_t_plus_one = sequence[time_step]

                for state_i in states:
                    for state_j in states:
                        sum_zai[state_i][state_j] += (
                            alpha[time_step][state_i]*
                            self.transition_probs[state_i][state_j]*
                            self.emission_probs[state_j][emission_at_t_plus_one]*
                            beta[time_step+1][state_j]
                            )/joint_prob
            
            # Also add zai values for the transition from the last state (at t=n) to STOP
            for state in states:
                sum_zai[state]["STOP"] += (
                    alpha[len(sequence)][state] * 
                    self.transition_probs[state]["STOP"]
                )/joint_prob
            
            return sum_zai



        def compute_gamma(sequence, alpha, beta):
            """
            Takes in a sequence and returns a dictionary, gamma[j][t],
            which represents the probability that the state at time step t is j, given the sequence of observations
            """
            states = self.transition_probs.keys()
            gamma = { state: { index : None for index in range(1, len(sequence) + 1 )} for state in states  }
            
            for time_step in range(1,len(sequence) + 1):
                sum_alpha_x_beta = sum (
                    [ alpha[time_step][state]*beta[time_step][state] for state in states ]
                )
            
                for state in states:
                    alpha_x_beta = alpha[time_step][state]*beta[time_step][state]
                    gamma[state][time_step] = alpha_x_beta / sum_alpha_x_beta
            
            return gamma

        def log_prob_of_seeing_sequence(sequence, alpha):
            """
            Computes log ( P (x1, x2, .... xn) )
            """
            prob_of_seeing_sequence = sum([ alpha[len(sequence)][state] * self.transition_probs[state]["STOP"] for state in self.transition_probs])
            
            # Take the logarithm of the probability, and return it
            return math.log(prob_of_seeing_sequence)
    
        # Compute alpha and beta
        alpha = forward_probabilities(sequence)
        beta = backward_probabilities(sequence)

        # Use computed values of alpha and beta to compute gamma and zai
        gamma = compute_gamma(sequence, alpha, beta)
        zai = compute_zai(sequence, alpha, beta)

        # Accumulate zai for START to state in time step 1
        for state in self.possible_states:
            accumulated_start_zai[state] += gamma[state][1]

        # Accumulate zai
        for state_i in self.possible_states:
            for state_j in self.possible_states + ["STOP"]:
                accumulated_zai[state_i][state_j] += zai[state_i][state_j]

        # Accumulate gamma
        for state in self.possible_states:
            for index,item in enumerate(sequence):
                accumulated_gamma[state][item] += gamma[state][index+1]
        
        return log_prob_of_seeing_sequence(sequence,alpha)

    def transition_probs_df(self):
        """
        Returns transition probabilities as a dataframe
        """
        i = []
        j = []
        prob = []

        for state in self.pi:
            i.append("START")
            j.append(state)
            prob.append(self.pi[state])

        for state_i in self.transition_probs:
            for state_j in self.transition_probs[state_i]:
                i.append(state_i)
                j.append(state_j)
                prob.append(self.transition_probs[state_i][state_j])
        
        return pd.DataFrame({"i":i, "j": j, "prob":prob})

    def emission_probs_df(self):
        """
        Returns emission probabilities as a dataframe
        """
        i = []
        emission = []
        prob = []

        for state in self.emission_probs:
            for em in self.emission_probs[state]:
                i.append(state)
                emission.append(em)
                prob.append(self.emission_probs[state][em])

        return pd.DataFrame({"i":i, "emission": emission, "prob":prob})

    # Question 2a (Naive Approach)
    @classmethod
    def compute_output_probs(cls, in_train_filename, output_probs_filename, delta=0.1):
        """
        Generates the P(x = w | y = j) for all w and j, saving them into a file called naive_output_probs.txt
        :param in_train_filename : File path to twitter_train.txt, which contains labelled POS tags for each word in tweets
        :param output_probs_filename: File path of naive_output_probs.txt
        :param delta: Smoothing parameter, typical values are 0.01, 0.1, 1, or 10
        """
        parsed_data = HMM.parse_twitter_training_data(in_train_filename)
        tag_dict = {}
        for tweet in parsed_data:  # Go through each tweet
            for word_and_tag in tweet:  # Go through each word in the tweet
                tag = word_and_tag[1]
                word = word_and_tag[0]
                # Add the word if it doesn't exist in the dict yet
                if tag not in tag_dict:
                    tag_dict[tag] = {"occurences": 1, "words": {word: 1}}
                else:
                    tag_dict[tag]["occurences"] += 1
                    # Add the tag if it doesn't exist in the dict yet
                    if word not in tag_dict[tag]["words"]:
                        tag_dict[tag]["words"][word] = 1
                    else:
                        tag_dict[tag]["words"][word] += 1

        # Calculate unique number of words
        num_words = len({word for tag in tag_dict.values() for word in tag["words"]})

        # Add probabilities for unseen data and apply smoothing
        for tag in tag_dict:
            count_y_is_j = tag_dict[tag]["occurences"]
            tag_dict[tag]["words"]["NONE"] = delta / (count_y_is_j + delta * (num_words + 1))
            for word in tag_dict[tag]["words"]:
                count_y_is_j_and_x_is_w = tag_dict[tag]["words"][word]

                # Compute the MLE estimate for this tag, for this word
                b_j_w = (count_y_is_j_and_x_is_w + delta) / (
                    count_y_is_j + delta * (num_words + 1)
                )
                tag_dict[tag]["words"][word] = b_j_w

        # Create output_dict, where tag, word and prob keys will each represent a column in the output dataframe
        output_dict = {"tag": [], "word": [], "prob": []}
        for tag in tag_dict:
            for word in tag_dict[tag]["words"]:
                b_j_w = tag_dict[tag]["words"][word]
                output_dict["tag"].append(tag)
                output_dict["word"].append(word)
                output_dict["prob"].append(b_j_w)

        # Convert output dict into a dataframe
        output_probs = pd.DataFrame.from_dict(output_dict)

        # Save dataframe to output prob file
        output_probs.to_csv(output_probs_filename, index=False)

    @classmethod
    def compute_output_probs_v2 (cls, in_train_filename, output_probs_filename):
        parsed_data = HMM.parse_twitter_training_data(in_train_filename)
        tag_dict = {}
        for tweet in parsed_data:  # Go through each tweet
            for word_and_tag in tweet:  # Go through each word in the tweet
                tag = word_and_tag[1]
                word = word_and_tag[0]
                # Add the word if it doesn't exist in the dict yet
                if tag not in tag_dict:
                    tag_dict[tag] = {"occurences": 1, "words": {word: 1}}
                else:
                    tag_dict[tag]["occurences"] += 1
                    # Add the tag if it doesn't exist in the dict yet
                    if word not in tag_dict[tag]["words"]:
                        tag_dict[tag]["words"][word] = 1
                    else:
                        tag_dict[tag]["words"][word] += 1

        def get_number_of_words_seen_r_times_in_tag(tag,r):
            words = tag_dict[tag]["words"]
            number = 0
            for word in words:
                count = words[word]
                if count == r:
                    number+=1
            return number

        # Compute r*, which is the good-turing smoothed value of r (ie. counts)
        # Number of words seen r times is nr
        def compute_r_star(r, number_of_words_seen_r_plus_one_times,number_of_words_seen_r_times):
            return (r+1) * ( number_of_words_seen_r_plus_one_times/number_of_words_seen_r_times )

        import copy
        smoothed_tag_dict = copy.deepcopy(tag_dict)
        for tag in tag_dict:
            tag_occurences = tag_dict[tag]["occurences"]
            words = tag_dict[tag]["words"]

            # All other n grams
            total_n_grams = 0
            for tag2 in tag_dict:
                if tag2 != tag:
                    total_n_grams += sum(tag_dict[tag2]["words"].values())

            # Smooth for unseen words using good-turing smoothing
            smoothed_tag_dict[tag]["words"]["NONE"] = get_number_of_words_seen_r_times_in_tag(tag,1)/total_n_grams

            sum_of_katz_counts = sum([count for count in smoothed_tag_dict[tag]["words"].values() ])
            smoothed_tag_dict[tag]["occurences"] = sum_of_katz_counts



        # Create output_dict, where tag, word and prob keys will each represent a column in the output dataframe
        output_dict = {"tag": [], "word": [], "prob": []}
        for tag in smoothed_tag_dict:
            count_y_is_j = smoothed_tag_dict[tag]["occurences"]
            for word in smoothed_tag_dict[tag]["words"]:
                count_y_is_j_and_x_is_w = smoothed_tag_dict[tag]["words"][word]

                # Compute the MLE estimate for this tag, for this word
                b_j_w = count_y_is_j_and_x_is_w /count_y_is_j
                output_dict["tag"].append(tag)
                output_dict["word"].append(word)
                output_dict["prob"].append(b_j_w)

        # Convert output dict into a dataframe
        output_probs = pd.DataFrame.from_dict(output_dict)

        # Save dataframe to output prob file
        output_probs.to_csv(output_probs_filename, index=False)


    @classmethod
    def compute_transition_probs(cls, in_train_filename, trans_probs_filename, in_tags_filename, sigma):
        """
        Generates the P(yt = j | yt-1 = i) for all w and j, saving them into a file called trans_probs.txt
        :param in_train_filename : File path to twitter_train.txt, which contains labelled POS tags for each word in tweets
        :param trans_probs_filename: File path of trans_probs.txt
        """
        # Aggregate all occurences in a df first with duplicates
        parsed_data = HMM.parse_twitter_training_data(in_train_filename)
        transition_df = pd.DataFrame(columns = ["Prior", "Next"])
        for tweet in parsed_data:  # Go through each tweet
            lenTweet = len(tweet) - 1 #Start counting from 0
            for index in range(len(tweet)):  # Go through each word in the tweet
                tag = tweet[index][1]
                # START
                if index == 0:
                    tempDF = pd.DataFrame.from_dict({"Prior": ["START"], "Next": [tag]})
                    transition_df = transition_df.append(tempDF, ignore_index=True)
                # END
                elif index == lenTweet:
                    tempDF = pd.DataFrame.from_dict({"Prior": [tag], "Next": ["END"]})
                    transition_df = transition_df.append(tempDF, ignore_index=True)
                # Other Cases
                else:
                    prev_tag = tweet[index-1][1]
                    tempDF = pd.DataFrame.from_dict({"Prior": [prev_tag], "Next": [tag]})
                    transition_df = transition_df.append(tempDF, ignore_index=True)
        
        # Sum up all occurences
        # Prior + Next
        transition_df = transition_df.groupby(['Prior', 'Next']).size().reset_index()
        transition_df.columns = ["Prior", "Next", "Count Tags"]
        piror_unique_tags = transition_df["Prior"].unique()
        # Prior Only
        prior_df = transition_df.drop(columns = "Next")
        prior_df = transition_df.groupby(['Prior']).size().to_frame('Count Tag')
        prior_df = prior_df.reset_index()

        # Get count of all possible unique tags
        parsed_tag_data = cls.parse_tags_data(in_tags_filename)
        parsed_tag_data_Start = parsed_tag_data + ["START"]
        parsed_tag_data_End = parsed_tag_data + ["END"]
        allUniqueTags = len(parsed_tag_data) + 2 # For END and START

        # Get probability calculated
        denominator = sigma*(allUniqueTags+1)
        transition_df_prob = transition_df.copy()
        transition_df_prob = transition_df.groupby(["Prior", "Next"])['Count Tags'].sum().rename("prob")
        transition_df_prob = (transition_df_prob + sigma) / (transition_df_prob.groupby(level=0).sum() + denominator)
        transition_df_prob.columns = ["Prior", "Next", "prob"]
        transition_df_prob = transition_df_prob.reset_index()
        
        # Find out what tags are already in transition_df and what not
        for i in parsed_tag_data_Start:
            # If tag does not exist in training data
            if i not in piror_unique_tags:
                # Create corresponding prob value in tag
                for j in parsed_tag_data_End:
                    # transition prob = 0 as tag did not appear in training data
                    # count of prior tag = 0 as tag did not appear in training data
                    tag_prob = sigma / denominator
                    corres_prob_df = pd.DataFrame.from_dict({"Prior": [i], "Next": [j], "prob": [tag_prob]})
                    transition_df_prob = transition_df_prob.append(corres_prob_df, ignore_index = True)
            # Get unique Next tags in each Prior tag
            next_unique_tags = transition_df[transition_df["Prior"] == i]["Next"].unique()
            if i != "START":
                parsed_data_tag_enu = parsed_tag_data_End
            elif i == "START":
                parsed_data_tag_enu = parsed_tag_data
            for j in parsed_data_tag_enu:
                # If tag does not exist in Next tag
                if j not in next_unique_tags:
                    # Get number of times tag Prior occured
                    prior_occured = prior_df[prior_df["Prior"] == i].iloc[0]["Count Tag"]
                    # transition prob = 0 as tag did not appear in training data
                    # count of prior tag = prior_occured
                    tag_prob = sigma / (prior_occured + denominator)
                    corres_prob_df = pd.DataFrame.from_dict({"Prior": [i], "Next": [j], "prob": [tag_prob]})
                    transition_df_prob = transition_df_prob.append(corres_prob_df, ignore_index = True)

        transition_df_prob.to_csv(trans_probs_filename, index=None)
    
        return transition_df_prob

    @classmethod
    def parse_tags_data(cls, in_tags_filename):
        """
        :param in_tags_filename : File name of twitter_tags.txt, which contains all possible tags
        :returns parsed_data: A list of all possible tags
        """
        parsed_data = []
        with open(in_tags_filename, "r") as f:
            for line in f:
                data = line[:-1] # Remove the newline character
                parsed_data += [data]

        return parsed_data

    @classmethod
    def parse_twitter_training_data(cls, in_train_filename):
        """
        :param in_train_filename : File name of twitter_train.txt, which contains labelled POS tags for each word in tweets
        :returns parsed_data: A list of lists containing each of the tweets and their respective words and tags
        """
        parsed_data = []
        current_tweet = []
        with open(in_train_filename, "r") as f:
            for line in f:
                if line == "\n":  # Start a new tweet if it's a blank line
                    parsed_data.append(current_tweet)
                    current_tweet = []
                else:  # Append data to current tweet
                    data = line.split("\t") #Split into word and tag
                    data[1] = data[1][:-1] # Remove the newline character
                    current_tweet.append(tuple(data))

        return parsed_data

    @classmethod
    def parse_data_no_tag(cls, data_no_tag):
        """
        :param data_no_tag : File name of data without tags, with each sequence seperated by spaces
        :returns parsed_data: A list of lists containing each of the sequences read from the file
        """
        parsed_data = []
        current_sequence = []
        with open(data_no_tag, "r") as f:
            for line in f:
                if line == "\n":  # Start a new sequence if it's a blank line
                    parsed_data.append(current_sequence)
                    current_sequence = []
                else:  # Append data to current sequence
                    current_sequence.append(line[:-1])

        return parsed_data

# Uncomment to generate naive_output_probs.txt
# in_train_filename = f"{ddir}/twitter_train.txt"
# naive_output_probs_filename = f"{ddir}/naive_output_probs.txt"
# HMM.compute_output_probs(in_train_filename, naive_output_probs_filename)

# Uncomment to generate viterbi2 output probs 
# in_train_filename = f"{ddir}/twitter_train.txt"
# output_probs_filename2 = f"{ddir}/output_probs2.txt"
# HMM.compute_output_probs_v2(in_train_filename,output_probs_filename2)

#%%

class Viterbi:

    @classmethod
    def compute_Prob_MLE(cls, in_train_filename, output_probs_filename, in_tags_filename, sigma):
        """
        For Q4 Part (a)
        Reuses Question 2 output probability function
        Transition Probability computed using freshly written code
        """

        sigma = 0.01
        # Reusing Question 2 function to compute output probability
        HMM.compute_output_probs(in_train_filename, output_probs_filename, delta=0.1)
        HMM.compute_transition_probs(in_train_filename, trans_probs_filename, in_tags_filename, sigma)
    
    @classmethod
    def dataframe_to_dict(cls, df, col1_name, col2_name):
        """
        Converts transition prob to a nested dict
        """
        from collections import defaultdict
        d = defaultdict(dict)

        for i, row in df.iterrows():
            d[str(row[col1_name])][str(row[col2_name])] = row.drop([col1_name, col2_name]).to_dict()

        dict_df = dict(d)

        return dict_df

    @classmethod
    def run_viterbi(cls,states,trans_prob,output_prob,sequence):
        """
        Given a sequence, the possible states, trans probs, and output probs, predicts tags for the sequence
        """

        def init_zeroes(states, trans_prob, output_prob, sequence):
            """
            Initiates Pi and Backpointer
            """
            # get START probabilities
            start_prob = trans_prob["START"]
            
            # Define first word
            firstWord = sequence[0]

            # Define statistics for pi and backpointer
            nLength = len(sequence)
            nState = len(states)

            # Define pi and Backpointer
            # Uses sequence + 1 length to account for END state
            pi = np.zeros(shape=(nLength, nState))
            backpointer = np.zeros(shape=(nLength, nState))

            # Iterate through states
            for i, state in enumerate(states):


                # get START -> state probability
                state_prob = start_prob[state]
                ao_v = state_prob["prob"]

                # get state -> output probability given word
                ## Check if word exists in output probability
                if firstWord in output_prob:
                    result_dict = output_prob[firstWord]
                else:
                    result_dict = output_prob["NONE"]

                if state in result_dict:
                    bv_x1 = result_dict[state]['prob']
                else:
                    result_dict = output_prob["NONE"]
                    bv_x1 = result_dict[state]['prob']

                # Calculate Prob
                prob = ao_v*bv_x1
                
                # Store in pi
                pi[0][i] = prob

            
            return [pi, backpointer]

        def compute_viterbi(states, trans_prob, output_prob, sequence, pi, backpointer):
            """
            Does the actual viterbi algorithm
            """

            def find_max(trans_prob, state, states, index, stateIndex, bv_xk, pi):
                """
                Finds arg max and max for each individual aij and pi[k]
                """
                # retrieve pi values
                pi_kminus1 = pi[index - 1]

                # set temp holder for results
                argMax = -1
                maxVal = -1

                # enumerate for u
                for priorIndex, prior in enumerate(states):

                    # get prior probabilities
                    prior_prob = trans_prob[prior]

                    # get prior -> state probability
                    state_prob = prior_prob[state]
                    au_v = state_prob["prob"]

                    # get previous pi
                    pi_kminus1_prior = pi_kminus1[priorIndex]

                    # calculate result
                    piResult = pi_kminus1_prior*au_v*bv_xk
                    
                    if piResult > maxVal:
                        maxVal = piResult
                        argMax = priorIndex

                return [maxVal, argMax]

            lastIndex = len(sequence) - 1

            for index, word in enumerate(sequence):

                ## Check if word exists in output probability
                if word in output_prob:
                    result_dict = output_prob[word]
                else:
                    result_dict = output_prob["NONE"]

                # START is covered in zero states
                if index != 0:
                    for stateIndex, state in enumerate(states):

                        # Check if state exists in word dict
                        if state in result_dict:
                            bv_xk = result_dict[state]['prob']
                        else:
                            result_dict_else = output_prob["NONE"]
                            bv_xk = result_dict_else[state]['prob']

                        # finding max and argmax
                        max_ArgMax_result = find_max(trans_prob, state, states, index, stateIndex, bv_xk, pi)
                        pi[index][stateIndex] = max_ArgMax_result[0]
                        backpointer[index][stateIndex] = max_ArgMax_result[1]

                    # ensure that probability does not go to zero for super long tweets
                    if all(i <= 0.00001 for i in pi[index]):
                        pi[index] = [i * 10000 for i in pi[index]]

            return [pi, backpointer]

        def getBackPointer(pi, backpointer, sequence, states):
            """
            Get backpointer results
            """
            # Get last state and index
            len_of_sequence = len(sequence)
            pi_list = pi[len_of_sequence-1]
            curr_index = np.argmax(pi_list)
            state_result = [states[curr_index]]
            path = [curr_index]
            prob_path = [pi[len_of_sequence-1][curr_index]]

            # access the relevant state
            for index in range(len_of_sequence-1, 0, -1):
                
                # Get index
                curr_index = int(backpointer[index][curr_index])

                # Get state
                state_result += [states[curr_index]]

                # Get path
                path += [curr_index]

                # Get prob
                prob_path += [pi[len_of_sequence-1][curr_index]]
            
            # reverse to get actual result
            list.reverse(state_result)
            list.reverse(path)
            list.reverse(prob_path)
            
            return [state_result, path, prob_path]

        # Initialise pi and backpointer, and compute results for START
        init_pi, init_backpointer = init_zeroes(states, trans_prob, output_prob, sequence)

        # Compute viterbi for the remaining sequence
        pi, backpointer = compute_viterbi(states, trans_prob, output_prob, sequence, init_pi, init_backpointer)
        
        # get the backpointer results, which is a tuple of 3 items: the state_result, the path, and the prob_path
        backpointer_result = getBackPointer(pi, backpointer, sequence, states)
        
        return backpointer_result
    
#%%
# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):

    # Read output probs from file
    output_probs = pd.read_csv(in_output_probs_filename)

    # Get all bj(w) assuming that count(y=j -> x=w) is 0 (ie there is no word w that exists with tag j in the training data)
    # We consider these the baseline probabilities, and we will update them if training data exists (ie. if there exists a word x that was tagged as tag j in the training data)
    # These "baseline" probabilities were stored as word == None in the output probs file, and the line below loads them as such
    baseline_probs = output_probs[output_probs["word"] == "NONE"].drop("word",axis=1).set_index("tag")

    # Read and parse test data from file
    test_data = HMM.parse_data_no_tag(in_test_filename)

    with open(out_prediction_filename, "w") as f:
        for tweet in test_data:
            for word in tweet:
                prob_for_each_tag = baseline_probs.copy()

                # Get output probs based on training data
                training_data_output_probs = output_probs[output_probs.word == word].set_index("tag")
                prob_for_each_tag.update(training_data_output_probs)

                # Get the tag that gives us the maximum probability for this word
                max_tag = prob_for_each_tag.idxmax().prob

                f.write(max_tag + "\n")

            f.write("\n")

#%%
def naive_predict2(
    in_output_probs_filename,
    in_train_filename,
    in_test_filename,
    out_prediction_filename,
):
    """
    P ( y=j | x=w ) = P( x=w | y=j ) * P( y=j ) / P( x=w )
    However, since we are only finding argmax, we can ignore the P( x=w ) term (the denominator)
    """

    # Read output probs from file
    output_probs = pd.read_csv(in_output_probs_filename)

    # Calculate number of unique words in output_probs
    num_words = len(output_probs.word.unique())

    # Get bj(w) assuming that count(y=j -> x=w) is 0 (ie there is no word w that exists with tag j in the training data)
    baseline_probs = output_probs[output_probs.word == "NONE"].drop("word",axis=1).set_index("tag")

    # Read and parse test data from file
    test_data = HMM.parse_data_no_tag(in_test_filename)

    # Generate a dictionary with tags being the keys and P( y=j ) being the values
    prob_y = {}
    parsed_data = HMM.parse_twitter_training_data(in_train_filename)
    for tweet in parsed_data:
        for word_and_tag in tweet:
            tag = word_and_tag[1]
            if tag not in prob_y:
                prob_y[tag] = 1
            else:
                prob_y[tag] += 1
    number_of_occurences = sum(prob_y.values())

    # Calculate P (y = j)
    prob_y = { k:v/number_of_occurences for (k,v) in prob_y.items() }

    with open(out_prediction_filename, "w") as f:
        for tweet in test_data:
            for word in tweet:
                prob_for_each_tag = baseline_probs.copy()

                # Update the probability of this word being of each tag using the output_probs previously generated 
                prob_for_each_tag.update(output_probs[output_probs.word == word].set_index("tag"))

                # Iterate through the probabilitiy of this word being of a particular tag, 
                # and update it, changing it to P( x=w | y=j ) * P( y=j )         
                for index, row in prob_for_each_tag.iterrows():
                    new_prob = row.prob * prob_y[index]
                    prob_for_each_tag.loc[index].prob = new_prob

                # Get the tag with the max probability, and write it to file
                max_tag = prob_for_each_tag.idxmax().prob
                f.write(max_tag + "\n")

            f.write("\n")

# Uncomment to generate output_probs.txt and trans_probs.txt
# in_train_filename = f"{ddir}/twitter_train.txt"
# output_probs_filename = f"{ddir}/output_probs.txt"
# trans_probs_filename = f"{ddir}/trans_probs.txt"
# in_tags_filename = f"{ddir}/twitter_tags.txt"
# Viterbi.compute_Prob_MLE(in_train_filename, output_probs_filename, in_tags_filename, 0.01)


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
   
    # Import the relevant files
    output_probability = pd.read_csv(in_output_probs_filename)
    transition_probability = pd.read_csv(in_trans_probs_filename)
    test_data = HMM.parse_data_no_tag(in_test_filename)
    states = HMM.parse_tags_data(in_tags_filename)

    # Convert transition and output probs to dict
    output_prob = Viterbi.dataframe_to_dict(output_probability,"word","tag")
    trans_prob = Viterbi.dataframe_to_dict(transition_probability,"Prior","Next")

    # Initialise 3 lists to save the results for each tweet in the test data
    state_result = []
    path = []
    prob_path = []

    # iterate through all tweets
    for tweet in test_data:

        viterbi_predictions = Viterbi.run_viterbi(states,trans_prob,output_prob,tweet)

        state_result += viterbi_predictions[0]
        path += viterbi_predictions[1]
        prob_path += viterbi_predictions[2]

    # Write predictions to file
    with open(out_predictions_filename, "w") as f:
        for prediction in state_result:
            f.write(prediction + "\n")
    
#%%

# Defunct, included here for documentation
class Viterbi2:
    
    def __init__():
        pass

    @classmethod
    def create_n_gram(cls, in_train_filename, in_tags_filename, trans_probs_filename2, sigma):
        """
        Generates n-gram given training dataset
        """
        # Aggregate all occurences in a df first with duplicates
        parsed_data = HMM.parse_twitter_training_data(in_train_filename)
        trigram = pd.DataFrame(columns = ["First", "Prior", "Curr"])

        for tweet in parsed_data:  # Go through each tweet
            lenTweet = len(tweet) - 1 #Start counting from 0
            if len(tweet) > 3:
                for index in range(len(tweet)):  # Go through each word in the tweet
                    tag = tweet[index][1]
                    
                    # START START ?
                    if index == 0:
                        prev_tag = tweet[index-1][1]
                        tempDF = pd.DataFrame.from_dict({"First": ["START"], "Prior": ["START"], "Curr": [tag]})
                        trigram = trigram.append(tempDF, ignore_index=True)

                    # START ? ?
                    elif index == 1:
                        prev_tag = tweet[index-1][1]
                        tempDF = pd.DataFrame.from_dict({"First": ["START"], "Prior": [prev_tag], "Curr": [tag]})
                        trigram = trigram.append(tempDF, ignore_index=True)

                    # ? ? END
                    elif index == lenTweet:
                        prev_tag = tweet[index-1][1]
                        tempDF = pd.DataFrame.from_dict({"First": [prev_tag], "Prior": [tag], "Curr": ["END"]})
                        trigram = trigram.append(tempDF, ignore_index=True)
                        
                    # ? ? ?
                    else:
                        prev_tag = tweet[index-1][1]
                        first_tag = tweet[index-2][1]
                        tempDF = pd.DataFrame.from_dict({"First": [first_tag], "Prior": [prev_tag], "Curr": [tag]})
                        trigram = trigram.append(tempDF, ignore_index=True)
            
            elif len(tweet) == 2:

                first_tag = tweet[0][1]
                second_tag = tweet[1][1]

                # START START First
                tempDF = pd.DataFrame.from_dict({"First": ["START"], "Prior": ["START"], "Curr": [first_tag]})
                trigram = trigram.append(tempDF, ignore_index=True)
                
                # START First Second
                tempDF = pd.DataFrame.from_dict({"First": ["START"], "Prior": [first_tag], "Curr": [second_tag]})
                trigram = trigram.append(tempDF, ignore_index=True)

                # First Second END
                tempDF = pd.DataFrame.from_dict({"First": [first_tag], "Prior": [second_tag], "Curr": ["END"]})
                trigram = trigram.append(tempDF, ignore_index=True)

                
            elif len(tweet) == 1:

                only_tag = tweet[0][1]
                # START START Only
                tempDF = pd.DataFrame.from_dict({"First": ["START"], "Prior": ["START"], "Curr": [only_tag]})
                trigram = trigram.append(tempDF, ignore_index=True)

                # START Only End
                tempDF = pd.DataFrame.from_dict({"First": ["START"], "Prior": [only_tag], "Curr": ["END"]})
                trigram = trigram.append(tempDF, ignore_index=True)

                
        #Get Trigram Prob
        trigram_prob = trigram.copy()
        trigram_prob = trigram_prob.groupby(["First", "Prior", "Curr"]).size().reset_index()
        trigram_prob.columns = ["First", "Prior", "Curr", "Count Tags_Triple"]
        trigram_prior = trigram.copy()
        trigram_prior = trigram_prior.groupby(["First", "Prior"]).size().reset_index()
        trigram_prior.columns = ["First", "Prior", "Count Tags_Left"]
        trigram_merged = pd.merge(trigram_prob, trigram_prior)
        # Apply Laplace Smoothing
        parsed_tag_data = HMM.parse_tags_data(in_tags_filename)
        allUniqueTags = len(parsed_tag_data) + 1 # For START
        denominator = sigma*(allUniqueTags+1)
        trigram_merged["prob"]  = (trigram_merged["Count Tags_Triple"] + sigma) / (trigram_merged["Count Tags_Left"] + denominator)
        trigram_merged = trigram_merged.drop(columns = ["Count Tags_Triple", "Count Tags_Left"])

        # Find out what tags are already in trigram and what not
        parsed_tag_data_Start = parsed_tag_data + ["START"]
        parsed_tag_data_End = parsed_tag_data + ["END"]
        first_unique_tags = trigram_merged["First"].unique()
        tag_prob = sigma / denominator

        # Iterate for First Tag
        for i in parsed_tag_data_Start:
            # If first tag does not exist in data, create all the tag data associated with i -> ? -> ?
            if i not in first_unique_tags:
                # Create corresponding prob value in tag
                for j in parsed_tag_data:

                    for k in parsed_tag_data_End:

                        corres_prob_df = pd.DataFrame.from_dict({"First": [i], "Prior": [j], "Curr": [k], "prob": [tag_prob]})
                        trigram_merged = trigram_merged.append(corres_prob_df, ignore_index = True)

            # Get unique Prior tags associated with each First tag
            prior_unique_tags = trigram_merged[trigram_merged["First"] == i]["Prior"].unique()
            
            # For each i -> diff j
            for j in parsed_tag_data_Start:
                # If prior tag does not exist in prior tag
                if j not in prior_unique_tags:
                    # Create corresponding prob value in tag associated with i -> j -> ?
                    for k in parsed_tag_data_End:
                        corres_prob_df = pd.DataFrame.from_dict({"First": [i], "Prior": [j], "Curr": [k], "prob": [tag_prob]})
                        trigram_merged = trigram_merged.append(corres_prob_df, ignore_index = True)
                
                elif j in prior_unique_tags:
                    # If prior tag exists, check for existence of k tag
                    curr_unique_tags = trigram_merged[trigram_merged["First"] == i][trigram_merged["Prior"] == j]["Curr"].unique()

                    for k in parsed_tag_data_End:
                        # Check if k exists in j
                        if k not in curr_unique_tags:
                            # Get number of times tag Prior occured
                            prior_occured = trigram_prior[trigram_prior["First"] == i][trigram_prior["Prior"] == j].iloc[0]["Count Tags_Left"]
                            # transition prob = 0 as tag did not appear in training data
                            # count of prior tag = prior_occured
                            tag_prob_tri = sigma / (prior_occured + denominator)
                            corres_prob_df = pd.DataFrame.from_dict({"First": [i], "Prior": [j], "Curr": [k], "prob": [tag_prob_tri]})
                            trigram_merged = trigram_merged.append(corres_prob_df, ignore_index = True)

        trigram_merged.to_csv(trans_probs_filename2, index=None)
        
        return trigram_merged

    @classmethod
    def dataframe_to_dict_trigram(cls, df, col1_name, col2_name, col3_name):
        """
        Converts transition prob to a nested dict: First, Prior, Curr
        """
        d = {}

        for index, row in df.iterrows():
            prob_value = row.drop([col1_name, col2_name, col3_name]).to_dict()
            first = str(row[col1_name])
            second = str(row[col2_name])
            third = str(row[col3_name])
            if first not in d:
                d.update({first: {}})
            if second not in d[first]:
                d[first].update({second: {}})
            if third not in d[first][second]:
                d[first][second].update({third: prob_value})

        dict_df = dict(d)

        return dict_df

    @classmethod
    def run_viterbi(cls,states,trans_prob,output_prob,sequence,trans_prob_curr):
        """
        Given a sequence, the possible states, trans probs, and output probs, predicts tags for the sequence
        """

        def init_zeroes(states, trans_prob, output_prob, sequence):
            """
            Initiates Pi and Backpointer
            """
            # get START->START probabilities
            start_prob = trans_prob["START"]["START"]
            
            # Define first word
            firstWord = sequence[0]

            # Define statistics for pi and backpointer
            nLength = len(sequence)
            nState = len(states)

            # Create temp storage for storing pi[V], where K is 0 (does not vary), V is diff states
            tempPi = np.zeros(shape=(nState))

            # Iterate through states
            for i, state in enumerate(states):

                # get START -> state probability
                state_prob = start_prob[state]
                ao_wv = state_prob["prob"]

                # get state -> output probability given word
                ## Check if word exists in output probability
                if firstWord in output_prob:
                    result_dict = output_prob[firstWord]
                else:
                    result_dict = output_prob["NONE"]

                if state in result_dict:
                    bv_x1 = result_dict[state]['prob']
                else:
                    result_dict = output_prob["NONE"]
                    bv_x1 = result_dict[state]['prob']

                # Calculate Prob
                prob = ao_wv*bv_x1
                
                # Store in pi as START -> START -> V -> Output(First word) == pi(0, W, U)
                tempPi[i] = prob


            ##### Calculate START -> U -> V

            # get START ->? -> ? probabilities
            start_prior_prob = trans_prob["START"]

            # Define second word
            secondWord = sequence[1]

            # Define pi and Backpointer
            # Shape is pi[K][U][V], U is prior, V is curr. tempPi[i] becomes prior, U
            pi = np.zeros(shape=(nLength, nState, nState))
            backpointer = np.zeros(shape=(nLength, nState, nState))

            # Get Start probabilities
            start_prob_second = trans_prob["START"]

            # Find values of pi[1]
            # Enumerate for V
            for v, statev in enumerate(states):

                # get state -> output probability given second word
                ## Check if word exists in output probability
                if secondWord in output_prob:
                    result_dict = output_prob[secondWord]
                else:
                    result_dict = output_prob["NONE"]

                if statev in result_dict:
                    bv_x1 = result_dict[statev]['prob']
                else:
                    result_dict = output_prob["NONE"]
                    bv_x1 = result_dict[statev]['prob']

                # Enumerate for U
                for u, stateu in enumerate(states):

                    # Since W is START for second word, can be accessed here
                    pi_kminus1_wu = tempPi[u]

                    # To find Max
                    piResult = -1
                    backpointerResult = -1

                    # Enumerate for W
                    for w, statew in enumerate(states):

                        # Get Start -> W -> U probabilities
                        start_prob_wu = start_prob_second[statew][stateu]["prob"]

                        calculatedResult = pi_kminus1_wu * start_prob_wu * bv_x1
                        
                        # The max function
                        if calculatedResult > piResult:
                            piResult = calculatedResult
                            backpointerResult = w

                    # After finding max, store in pi and backpointer
                    pi[1][u][v] = calculatedResult
                    backpointer[1][u][v] = backpointerResult

            return [pi, backpointer]

        def compute_viterbi(states, trans_prob, output_prob, sequence, pi, backpointer):
            """
            Does the actual viterbi algorithm
            """

            def find_max(trans_prob, k, v, statev, states, bv_xk, pi):
                """
                Finds arg max and max for each individual aij and pi[k]
                """
                # retrieve pi values array
                pi_kminus1 = pi[k - 1]

                # Enumerate for U:
                for u, stateu in enumerate(states):

                    # set temp holder for results
                    argMax = -1
                    maxVal = -1
                    
                    # Enumerate for W
                    for w, statew in enumerate(states):

                        # get first -> prior -> state probability
                        state_prob = trans_prob[statew][stateu][statev]
                        a_wu_v = state_prob["prob"]

                        # get previous pi
                        pi_kminus1_wu = pi_kminus1[w][u]
                        
                        # calculate result
                        piResult = pi_kminus1_wu * a_wu_v * bv_xk
                        
                        if piResult > maxVal:
                            maxVal = piResult
                            argMax = w

                    # After finding max, store in pi and backpointer
                    pi[k][u][v] = maxVal
                    backpointer[k][u][v] = argMax

                return [pi, backpointer]

            # for K (word)
            for index, word in enumerate(sequence):

                ## Check if word exists in output probability
                if word in output_prob:
                    result_dict = output_prob[word]
                else:
                    result_dict = output_prob["NONE"]

                # START -> START ->?, START -> U -> V is covered in zero states
                if index > 1:

                    # V (curr val)
                    for v, statev in enumerate(states):

                        # Check if state exists in word dict
                        if statev in result_dict:
                            bv_xk = result_dict[statev]['prob']
                        else:
                            result_dict_else = output_prob["NONE"]
                            bv_xk = result_dict_else[statev]['prob']

                        max_ArgMax_result = find_max(trans_prob, index, v, statev, states, bv_xk, pi)
                        pi = max_ArgMax_result[0]
                        backpointer = max_ArgMax_result[1]

            return [pi, backpointer]

        
        # Needs Fix
        def getBackPointer(pi, backpointer, sequence, states, trans_prob):
            """
            Get backpointer results
            """
            # Get last state and index
            seqLength = len(sequence)-1
            
            # To find y(n-1) and y(n) first
            lastPi = pi[seqLength]
            # Find argmax (u, v)
            largestValue = -1
            largest_u = -1
            largest_stateu = -1
            largest_v = -1
            largest_statev = -1

            # Get the END state probability
            #rob_END_Array = trans_prob["END"]

            # Enumerate to find the last first
            for u, stateu in enumerate(states):

                lastPi_u = lastPi[u]
                #u_END_piValue = prob_END_Array[stateu]

                # Then find the prior
                for v, statev in enumerate(states):
                    
                    lastPi_uv = lastPi_u[v]
                    #uv_END_piValue = u_END_piValue[statev]["Prob"]

                    #lastPi_uv_END = lastPi_uv * uv_END_piValue

                    lastPi_uv_END = lastPi_uv

                    # The max function
                    if lastPi_uv_END > largestValue:
                        # Keep track of prob
                        largestValue = lastPi_uv_END
                        # Keep track of u
                        largest_u = u
                        largest_stateu = stateu
                        # Keep track of v
                        largest_v= v
                        largest_statev = statev

            # Tracker for the result
            prob_path = [largestValue]
            path = [largest_v, largest_u] # For reversal later, stores index
            statePath = [largest_statev, largest_stateu] # For reversal later

            # For running through the rest
            index_u = largest_u
            index_v = largest_v

            # access the relevant state
            for index in range(seqLength, 1, -1):
                
                # Get access to the backpointer array, probability array for the index
                backpointerArray = backpointer[index]
                probArray = pi[index]

                # Get the index to the previous state, given W -> U -> V
                index_w = int(backpointerArray[index_u][largest_v])

                # Get state for W and store it
                statePath += [states[index_w]]

                # Get path
                path += [index_w]

                # Get prob
                prob_path += [probArray[index_w][index_u]]

                # First index becomes prior, Prior index becomes curr
                index_w, index_u = index_u, index_v
            
            # reverse to get actual result
            list.reverse(statePath)
            list.reverse(path)
            list.reverse(prob_path)
            
            return [statePath, path, prob_path]
            

        # Initialise pi and backpointer, and compute results for START 
        init_pi, init_backpointer = init_zeroes(states, trans_prob, output_prob, sequence)

        # Compute viterbi for the remaining sequence
        pi, backpointer = compute_viterbi(states, trans_prob, output_prob, sequence, init_pi, init_backpointer)
        
        # get the backpointer results, which is a tuple of 3 items: the state_result, the path, and the prob_path
        backpointer_result = getBackPointer(pi, backpointer, sequence, states, trans_prob_curr)
        
        return backpointer_result



#%%
# Defunct: Calculate Trigram Probability and export as CSV
# Uncomment to generate the trans_probs2.txt file
# in_train_filename = f"{ddir}/twitter_train.txt"
# in_tags_filename = f"{ddir}/twitter_tags.txt"
# trans_probs_filename2 = f"{ddir}/trans_probs2.txt"
# sigma = 0.01
# trigram_data = Viterbi2.create_n_gram(in_train_filename, in_tags_filename, trans_probs_filename2, sigma)

#%%
def viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename, viterbi_predictions_filename2):
    
    # Import the relevant files
    output_probability = pd.read_csv(output_probs_filename2)
    transition_probability = pd.read_csv(trans_probs_filename2)
    test_data = HMM.parse_data_no_tag(in_test_filename)
    states = HMM.parse_tags_data(in_tags_filename)

    # Convert transition and output probs to dict
    # Handle the two different probability formats we have
    if "word" in output_probability.columns:
        output_prob = Viterbi.dataframe_to_dict(output_probability,"word","tag")
        trans_prob = Viterbi.dataframe_to_dict(transition_probability,"Prior","Next")
    else:
        output_prob = Viterbi.dataframe_to_dict(output_probability,"emission","i")
        trans_prob = Viterbi.dataframe_to_dict(transition_probability,"i","j")

    # Initialise 3 lists to save the results for each tweet in the test data
    state_result = []
    path = []
    prob_path = []

    # iterate through all tweets
    for tweet in test_data:

        viterbi_predictions = Viterbi.run_viterbi(states,trans_prob,output_prob,tweet)

        state_result += viterbi_predictions[0]
        path += viterbi_predictions[1]
        prob_path += viterbi_predictions[2]

    # Write predictions to file
    with open(viterbi_predictions_filename2, "w") as f:
        for prediction in state_result:
            f.write(prediction + "\n")


# For Question 6b - uncomment to generate output_probs3 and trans_probs3
# in_tag_filename = f"{ddir}/twitter_tags.txt"
# in_train_filename = f"{ddir}/twitter_train_no_tag.txt"
# trans_probs_filename3 = f"{ddir}/trans_probs3.txt"
# output_probs_filename3 = f"{ddir}/output_probs3.txt"
# def forward_backward_no_train(
#     in_train_filename,
#     in_tag_filename,
#     out_trans_filename,
#     out_output_filename,
#     seed,
# ):
#     # Get all potential tags from file
#     potential_tags = []
#     with open(in_tag_filename,"r") as f:
#         for tag in f.readlines():
#             potential_tags.append(tag[:-1])

#     # Get all unique possible emissions (words) as given in the training data
#     training_data = HMM.parse_data_no_tag(in_train_filename)
#     unique_emissions = list({ emission for sequence in training_data for emission in sequence })

#     # Initialise our hidden markov model
#     hmm = HMM(potential_tags,unique_emissions,seed)

#     # Save transition and emission probs to file
#     hmm.transition_probs_df().to_csv(out_trans_filename, index=False)
#     hmm.emission_probs_df().to_csv(out_output_filename, index=False)
# forward_backward_no_train(in_train_filename,in_tag_filename,trans_probs_filename3,output_probs_filename3,8)


#%%
def forward_backward(
    in_train_filename,
    in_tag_filename,
    out_trans_filename,
    out_output_filename,
    max_iter,
    seed,
    thresh,
):
    # Get all potential tags from file
    potential_tags = []
    with open(in_tag_filename,"r") as f:
        for tag in f.readlines():
            potential_tags.append(tag[:-1])

    # Get all unique possible emissions (words) as given in the training data
    training_data = HMM.parse_data_no_tag(in_train_filename)
    unique_emissions = list({ emission for sequence in training_data for emission in sequence })

    # Initialise our hidden markov model
    hmm = HMM(potential_tags,unique_emissions,seed)

    # Run forward backward, breaking when max_iter or threshold is reached
    hmm.forward_backward_algorithm(training_data,max_iter,thresh)

    # Save transition and emission probs to file
    hmm.transition_probs_df().to_csv(out_trans_filename, index=False)
    hmm.emission_probs_df().to_csv(out_output_filename, index=False)


def cat_predict(
    in_test_filename,
    in_trans_probs_filename,
    in_output_probs_filename,
    in_states_filename,
    out_predictions_file,
):
    # Import the relevant files
    output_probability = pd.read_csv(in_output_probs_filename)
    transition_probability = pd.read_csv(in_trans_probs_filename)
    test_data = HMM.parse_data_no_tag(in_test_filename)
    states = HMM.parse_tags_data(in_states_filename)

    # Convert output and transition probs to dict
    output_prob = Viterbi.dataframe_to_dict(output_probability,"emission","i")
    trans_prob = Viterbi.dataframe_to_dict(transition_probability, "i", "j")

    # Each item in the predictions list represents the prediction for each sequence
    predictions = []

    # Iterate through all test data
    for sequence in test_data:

        viterbi_predictions = Viterbi.run_viterbi(states,trans_prob,output_prob,sequence)
        state_predictions = viterbi_predictions[0]
        state_prediction_of_last_item = state_predictions[-1]

        # We multiply the probability of transition from the nth state to the n+1th state and the output probability of that state
        # To obtain the most likely output
        most_likely_output = None
        prob_of_most_likely_output = 0
        for state in trans_prob[state_prediction_of_last_item]:
            if state == "STOP":
                continue
            state_prob = trans_prob[state_prediction_of_last_item][state]["prob"]
            for output in output_prob:
                prob = output_prob[output][state]["prob"] * state_prob
                if prob > prob_of_most_likely_output:
                    most_likely_output = output
                    prob_of_most_likely_output = prob

        # Add the price change prediction to the predictions list
        predictions.append(most_likely_output)
    
    # Write to file
    with open(out_predictions_file, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")
    

#%%
def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth:
            correct += 1
    return correct, len(predicted_tags), correct / len(predicted_tags)


def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth)) ** 2
    return error / len(predicted_tags), error, len(predicted_tags)

#%%
def run():
    """
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    """

    ddir = "/bt3102project/submission"

    in_train_filename = f"{ddir}/twitter_train.txt"

    naive_output_probs_filename = f"{ddir}/naive_output_probs.txt"

    in_test_filename = f"{ddir}/twitter_dev_no_tag.txt"
    in_ans_filename = f"{ddir}/twitter_dev_ans.txt"
    naive_prediction_filename = f"{ddir}/naive_predictions.txt"
    
    naive_predict(
        naive_output_probs_filename, in_test_filename, naive_prediction_filename
    )
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f"Naive prediction accuracy:     {correct}/{total} = {acc}")

    
    naive_prediction_filename2 = f"{ddir}/naive_predictions2.txt"
    naive_predict2(
        naive_output_probs_filename,
        in_train_filename,
        in_test_filename,
        naive_prediction_filename2,
    )
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f"Naive prediction2 accuracy:    {correct}/{total} = {acc}")

    

    trans_probs_filename = f"{ddir}/trans_probs.txt"
    output_probs_filename = f"{ddir}/output_probs.txt"

    in_tags_filename = f"{ddir}/twitter_tags.txt"
    viterbi_predictions_filename = f"{ddir}/viterbi_predictions.txt"
    
    viterbi_predict(in_tags_filename,
        trans_probs_filename,
        output_probs_filename,
        in_test_filename,
        viterbi_predictions_filename,
    )
    
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f"Viterbi prediction accuracy:   {correct}/{total} = {acc}")
    
    trans_probs_filename2 = f"{ddir}/trans_probs2.txt"
    output_probs_filename2 = f"{ddir}/output_probs2.txt"
    
    viterbi_predictions_filename2 = f"{ddir}/viterbi_predictions2.txt"

    viterbi_predict2(
        in_tags_filename,
        trans_probs_filename2,
        output_probs_filename2,
        in_test_filename,
        viterbi_predictions_filename2,
    )
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f"Viterbi2 prediction accuracy:  {correct}/{total} = {acc}")

    in_train_filename = f"{ddir}/twitter_train_no_tag.txt"
    in_tag_filename = f"{ddir}/twitter_tags.txt"
    out_trans_filename = f"{ddir}/trans_probs4.txt"
    out_output_filename = f"{ddir}/output_probs4.txt"
    max_iter = 10
    seed = 8
    thresh = 1e-4
    forward_backward(
        in_train_filename,
        in_tag_filename,
        out_trans_filename,
        out_output_filename,
        max_iter,
        seed,
        thresh,
    )
    in_tags_filename = f"{ddir}/twitter_tags.txt"
    trans_probs_filename3 = f"{ddir}/trans_probs3.txt"
    output_probs_filename3 = f"{ddir}/output_probs3.txt"
    viterbi_predictions_filename3 = f"{ddir}/fb_predictions3.txt"
    viterbi_predict2(
        in_tags_filename,
        trans_probs_filename3,
        output_probs_filename3,
        in_test_filename,
        viterbi_predictions_filename3,
    )
    correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    print(f"iter 0 prediction accuracy:    {correct}/{total} = {acc}")

    trans_probs_filename4 = f"{ddir}/trans_probs4.txt"
    output_probs_filename4 = f"{ddir}/output_probs4.txt"
    viterbi_predictions_filename4 = f"{ddir}/fb_predictions4.txt"
    viterbi_predict2(
        in_tags_filename,
        trans_probs_filename4,
        output_probs_filename4,
        in_test_filename,
        viterbi_predictions_filename4,
    )
    correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    print(f"iter 10 prediction accuracy:   {correct}/{total} = {acc}")

    in_train_filename = f"{ddir}/cat_price_changes_train.txt"
    in_tag_filename = f"{ddir}/cat_states.txt"
    out_trans_filename = f"{ddir}/cat_trans_probs.txt"
    out_output_filename = f"{ddir}/cat_output_probs.txt"
    max_iter = 1000000
    seed = 8
    thresh = 1e-4
    
    forward_backward(
        in_train_filename,
        in_tag_filename,
        out_trans_filename,
        out_output_filename,
        max_iter,
        seed,
        thresh,
    )

    in_test_filename = f"{ddir}/cat_price_changes_dev.txt"
    in_trans_probs_filename = f"{ddir}/cat_trans_probs.txt"
    in_output_probs_filename = f"{ddir}/cat_output_probs.txt"
    in_states_filename = f"{ddir}/cat_states.txt"
    predictions_filename = f"{ddir}/cat_predictions.txt"
    cat_predict(
        in_test_filename,
        in_trans_probs_filename,
        in_output_probs_filename,
        in_states_filename,
        predictions_filename,
    )

    in_ans_filename = f"{ddir}/cat_price_changes_dev_ans.txt"
    ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(
        predictions_filename, in_ans_filename
    )
    print(f"average squared error for {num_ex} examples: {ave_sq_err}")


if __name__ == "__main__":
    run()


#%%

