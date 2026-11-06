import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "eval.py"


class TrackedModule(ModuleType):
    def __init__(self, name, **values):
        super().__init__(name)
        self.values = values
        self.accesses = []

    def __getattr__(self, name):
        if name in self.values:
            self.accesses.append(name)
            return self.values[name]
        raise AttributeError(name)


class EvalEntrypointTests(unittest.TestCase):
    def run_fixture(self, run_name):
        fake_train = ["synthetic example"]
        fake_agent = Mock()
        agent = TrackedModule("agent", Agent=fake_agent)
        preprocess = TrackedModule("preprocess", train=fake_train, test=[])
        optimizer = Mock()
        dspy = SimpleNamespace(
            Signature=object, InputField=Mock(), OutputField=Mock(), MIPROv2=Mock(return_value=optimizer),
        )
        with patch.dict(sys.modules, {
            "dspy": dspy,
            "dspy.evaluate": SimpleNamespace(Evaluate=Mock()),
            "agent": agent,
            "preprocess": preprocess,
        }):
            module = runpy.run_path(str(SCRIPT), run_name=run_name)
        return module, agent, preprocess, dspy, optimizer

    def test_import_does_not_load_agent_data_or_run_optimizer(self):
        module, agent, preprocess, dspy, _ = self.run_fixture("fixture_eval")
        self.assertTrue(callable(module["metric"]))
        self.assertTrue(callable(module["evaluate_agent"]))
        self.assertEqual(agent.accesses, [])
        self.assertEqual(preprocess.accesses, [])
        dspy.MIPROv2.assert_not_called()

    def test_explicit_execution_still_runs_optimizer(self):
        module, agent, preprocess, dspy, optimizer = self.run_fixture("__main__")
        dspy.MIPROv2.assert_called_once_with(metric=module["metric"])
        optimizer.compile.assert_called_once_with(
            agent.values["Agent"].return_value, trainset=preprocess.values["train"],
        )
        optimizer.compile.return_value.save.assert_called_once_with(path="/tmp/model.json")
