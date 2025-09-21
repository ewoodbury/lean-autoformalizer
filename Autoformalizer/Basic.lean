namespace Autoformalizer

/-- Example lemma used to verify the Lean environment builds successfully. -/
theorem add_comm_example (a b : Nat) : a + b = b + a :=
  Nat.add_comm a b

end Autoformalizer
