"""Interface code for rllab. Mainly handles interaction with mdpsim & hard
things like action masking."""

from warnings import warn

import numpy as np
from typing import Iterable, List, Optional, Tuple
from typing_extensions import Self
from util.prob_dom_meta import BoundProp, BoundAction


def strip_parens(thing):
    """Convert string of form `(foo bar baz)` to `foo bar baz` (i.e. strip
    leading & trailing parens). More complicated than it should be b/c it does
    safety checks to catch my bugs :)"""
    assert len(thing) > 2 and thing[0] == "(" and thing[-1] == ")", \
        "'%s' does not look like it's surrounded by parens" % (thing,)
    stripped = thing[1:-1]
    assert "(" not in stripped and ")" not in stripped, \
        "parens in '%s' aren't limited to start and end" % (thing,)
    return stripped

class CanonicalState(object):
    """The ASNet code uses a lot of state representations. There are
    pure-Python state representations, there are state representations based on
    the SSiPP & MDPsim wrappers, and there are string-based intermediate
    representations used to convert between the other representations. This
    class aims to be a single canonical state class that it is:

    1. Possible to convert to any other representation,
    2. Possible to instantiate from any other representation,
    3. Possible to pickle & send between processes.
    4. Efficient to manipulate.
    5. Relatively light on memory."""

    def __init__(self,
                 bound_prop_truth: Iterable[Tuple[BoundProp, bool]],
                 bound_acts_enabled: Iterable[Tuple[BoundAction, bool]],
                 is_goal: bool):
        # note: props and acts should always use the same order! I don't want
        # to be passing around extra data to store "real" order for
        # propositions and actions all the time :(
        # FIXME: replace props_true and acts_enabled with numpy ndarray masks
        # instead of inefficient list-of-tuples structure
        self.props_true = tuple(bound_prop_truth)
        self.acts_enabled = tuple(bound_acts_enabled)
        self.is_goal = is_goal
        self.is_terminal = is_goal or not any(
            enabled for _, enabled in self.acts_enabled)
        self._aux_data = None
        self._aux_data_interp = None
        self._aux_data_interp_to_id = None


    def _do_validate(self):
        """Run some sanity checks on the newly-constructed state."""
        # first check proposition mask
        for prop_idx, prop_tup in enumerate(self.props_true):
            # should be tuple of (proposition, truth value)
            assert isinstance(prop_tup, tuple) and len(prop_tup) == 2
            assert isinstance(prop_tup[0], BoundProp)
            assert isinstance(prop_tup[1], bool)
            if prop_idx > 0:
                # should come after previous proposition alphabetically
                assert prop_tup[0].unique_ident \
                    > self.props_true[prop_idx - 1][0].unique_ident

        # next check action mask
        for act_idx, act_tup in enumerate(self.acts_enabled):
            # should be tuple of (action, enabled flag)
            assert isinstance(act_tup, tuple) and len(act_tup) == 2
            assert isinstance(act_tup[0], BoundAction)
            assert isinstance(act_tup[1], bool)
            if act_idx > 0:
                # should come after previous action alphabetically
                assert act_tup[0].unique_ident \
                    > self.acts_enabled[act_idx - 1][0].unique_ident

        # make sure that auxiliary data is 1D ndarray
        if self._aux_data is not None:
            assert isinstance(self._aux_data, np.ndarray), \
                "_aux_data is not ndarray (%r)" % type(self._aux_data)
            assert self._aux_data.ndim == 1

    def __repr__(self) -> str:
        """Return a string representation of the state.

        Returns:
            str: string representation of the state.
        """
        # Python-legible state
        return '%s(%r, %r)' \
            % (self.__class__.__name__, self.props_true, self.acts_enabled)

    def __str__(self) -> str:
        """Return a human-readable string representation of the state.

        Returns:
            str: human-readable string representation of the state.
        """
        # human-readable state
        prop_str = '), ('.join(p.unique_ident for p, t in self.props_true if t)
        prop_str = prop_str or '-'
        state_str = '(%s)' % (prop_str)
        return state_str

    def _ident_tup(self) \
            -> Tuple[Tuple[BoundProp, bool], Tuple[BoundAction, bool], bool]:
        """Return a tuple that uniquely identifies this state.

        Returns:
            Tuple[Tuple[BoundProp, bool], Tuple[BoundAction, bool], bool]: a
            tuple that uniquely identifies this state.
        """
        # This function is used to get a hashable representation for __hash__
        # and __eq__. Note that we don't hash _aux_data because it's an
        # ndarray; instead, hash bool indicating whether we have _aux_data.
        # Later on, we WILL compare on _aux_data in the __eq__ method.
        # (probably it's a bad idea not to include that in the hash, but
        # whatever)
        return (self.props_true, self.acts_enabled, self._aux_data is None)

    def __hash__(self) -> int:
        """Return a hash of the state.

        Returns:
            int: hash of the state.
        """
        return hash(self._ident_tup())

    def __eq__(self, other: Self) -> bool:
        """Return whether this state is equal to another state.

        Args:
            other (Self): state to compare against.

        Raises:
            TypeError: if other is not a CanonicalState.

        Returns:
            bool: whether this state is equal to other.
        """
        if not isinstance(other, CanonicalState):
            raise TypeError(
                "Can't compare self (type %s) against other object (type %s)" %
                (type(self), type(other)))
        eq_basic = self._ident_tup() == other._ident_tup()
        if self._aux_data is not None and eq_basic:
            # equality depends on _aux_data being similar/identical
            return np.allclose(self._aux_data, other._aux_data)
        return eq_basic


    ##################################################################
    # MDPSim interop routines
    ##################################################################

    @classmethod
    def from_mdpsim(cls,
                    mdpsim_state,
                    planner_exts,
                    *,
                    prev_cstate=None,
                    prev_act=None,
                    is_init_cstate=None):
        # general strategy: convert both props & actions to string repr, then
        # use those reprs to look up equivalent BoundProposition/BoundAction
        # representation from problem_meta
        problem_meta = planner_exts.problem_meta
        mdpsim_props_true \
            = planner_exts.mdpsim_problem.prop_truth_mask(mdpsim_state)
        truth_val_by_name = {
            # <mdpsim_prop>.identifier includes parens around it, which we want
            # to strip
            strip_parens(mp.identifier): truth_value
            for mp, truth_value in mdpsim_props_true
        }
        # now build mask from actual BoundPropositions in right order
        prop_mask = [(bp, truth_val_by_name[bp.unique_ident])
                     for bp in problem_meta.bound_props_ordered]

        # similar stuff for action selection
        mdpsim_acts_enabled \
            = planner_exts.mdpsim_problem.act_applicable_mask(mdpsim_state)
        act_on_by_name = {
            strip_parens(ma.identifier): enabled
            for ma, enabled in mdpsim_acts_enabled
        }
        act_mask = [(ba, act_on_by_name[ba.unique_ident])
                    for ba in problem_meta.bound_acts_ordered]

        is_goal = mdpsim_state.goal()

        return cls(prop_mask,
                   act_mask,
                   is_goal)

    def _to_state_string(self):
        # convert this state to a SSiPP-style state string
        ssipp_string = ', '.join(bp.unique_ident
                                 for bp, is_true in self.props_true if is_true)
        # XXX: remove this check once method tested
        assert ')' not in ssipp_string and '(' not in ssipp_string
        return ssipp_string

    def to_mdpsim(self, planner_exts):
        # yes, for some reason I originally made MDPSim take SSiPP-style
        # strings in this *one* place
        ssipp_style_string = self._to_state_string()
        problem = planner_exts.mdpsim_problem
        mdpsim_state = problem.intermediate_atom_state(ssipp_style_string)
        return mdpsim_state


    def to_frozen_tuple(self):
        return set([x.strip() for x in str(self).split(",")])



def get_action_name(planner_exts, action_id: int) \
        -> Optional[str]:
    """Returns the name of the action with the given ID.

    Returns:
        str | None: The name of the action, or None if the action ID is
        invalid.
    """
    acts_ordered = planner_exts.problem_meta.bound_acts_ordered
    if 0 <= action_id < len(acts_ordered):
        bound_act = acts_ordered[action_id]
        return bound_act.unique_ident
    return None  # explicit return b/c silent failure is intentional
