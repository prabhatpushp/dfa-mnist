import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union, Set


class Automaton(ABC):
    """
    Abstract base class for automata.
    
    An automaton is a mathematical model of computation that can be in one of a
    finite number of states at any given time. It can transition from one state
    to another in response to an input symbol.
    """
    
    @abstractmethod
    def process_input(self, input_sequence: Any) -> bool:
        """
        Process an input sequence and determine if it is accepted.
        
        Args:
            input_sequence: The input sequence to process.
            
        Returns:
            True if the input sequence is accepted, False otherwise.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the automaton to its initial state.
        """
        pass
    
    @abstractmethod
    def get_current_state(self) -> Any:
        """
        Get the current state of the automaton.
        
        Returns:
            The current state of the automaton.
        """
        pass
    
    @abstractmethod
    def is_in_final_state(self) -> bool:
        """
        Check if the automaton is in a final state.
        
        Returns:
            True if the automaton is in a final state, False otherwise.
        """
        pass


class FiniteAutomaton(Automaton):
    """
    Abstract base class for finite automata.
    
    A finite automaton is an automaton with a finite number of states.
    """
    
    @abstractmethod
    def get_num_states(self) -> int:
        """
        Get the number of states in the automaton.
        
        Returns:
            The number of states in the automaton.
        """
        pass
    
    @abstractmethod
    def get_alphabet(self) -> Set:
        """
        Get the alphabet of the automaton.
        
        Returns:
            The set of symbols in the alphabet.
        """
        pass
    
    @abstractmethod
    def get_transition_function(self) -> Dict:
        """
        Get the transition function of the automaton.
        
        Returns:
            The transition function as a dictionary.
        """
        pass
    
    @abstractmethod
    def get_initial_state(self) -> Any:
        """
        Get the initial state of the automaton.
        
        Returns:
            The initial state of the automaton.
        """
        pass
    
    @abstractmethod
    def get_final_states(self) -> Set:
        """
        Get the final states of the automaton.
        
        Returns:
            The set of final states of the automaton.
        """
        pass


class DeterministicFiniteAutomaton(FiniteAutomaton):
    """
    Abstract base class for deterministic finite automata (DFA).
    
    A DFA is a finite automaton that can be in exactly one state at any given time.
    For each state and input symbol, there is exactly one transition to a next state.
    """
    
    @abstractmethod
    def transition(self, input_symbol: Any) -> None:
        """
        Transition to the next state based on the input symbol.
        
        Args:
            input_symbol: The input symbol to process.
        """
        pass


class NonDeterministicFiniteAutomaton(FiniteAutomaton):
    """
    Abstract base class for non-deterministic finite automata (NFA).
    
    An NFA is a finite automaton that can be in multiple states at the same time.
    For each state and input symbol, there can be multiple transitions to different states.
    """
    
    @abstractmethod
    def get_current_states(self) -> Set:
        """
        Get the current states of the automaton.
        
        Returns:
            The set of current states of the automaton.
        """
        pass
    
    @abstractmethod
    def transition(self, input_symbol: Any) -> None:
        """
        Transition to the next states based on the input symbol.
        
        Args:
            input_symbol: The input symbol to process.
        """
        pass


class PushdownAutomaton(Automaton):
    """
    Abstract base class for pushdown automata (PDA).
    
    A PDA is an automaton that can use a stack to store information.
    It can push symbols onto the stack and pop symbols from the stack.
    """
    
    @abstractmethod
    def get_stack(self) -> List:
        """
        Get the current stack of the automaton.
        
        Returns:
            The current stack of the automaton.
        """
        pass
    
    @abstractmethod
    def push(self, symbol: Any) -> None:
        """
        Push a symbol onto the stack.
        
        Args:
            symbol: The symbol to push onto the stack.
        """
        pass
    
    @abstractmethod
    def pop(self) -> Any:
        """
        Pop a symbol from the stack.
        
        Returns:
            The symbol popped from the stack.
        """
        pass
    
    @abstractmethod
    def peek(self) -> Any:
        """
        Peek at the top symbol on the stack without removing it.
        
        Returns:
            The top symbol on the stack.
        """
        pass 