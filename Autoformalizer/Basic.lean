-- Basic imports for autoformalizer proofs
import Mathlib.Data.Set.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Bool.Basic
import Mathlib.Logic.Function.Basic

namespace Autoformalizer

/-- Example lemma used to verify the Lean environment builds successfully. -/
theorem add_comm_example (a b : Nat) : a + b = b + a :=
  Nat.add_comm a b

end Autoformalizer
