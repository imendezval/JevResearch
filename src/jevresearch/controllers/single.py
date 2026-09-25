"""Select the sole sampled proposal without a controller API call."""


class SingleCandidateController:
    kind = "single"
    selection_mode = "local"

    def select(self, state, candidates, rng_state):
        if len(candidates) != 1:
            raise ValueError("single-candidate selector requires one proposal")
        return candidates[0].id, rng_state
