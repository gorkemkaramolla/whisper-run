import onnxruntime as ort


def init_session(model_path):
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=opts)
    return sess


class PickableInferenceSession:
    """
    This is a wrapper to make the current InferenceSession class pickable.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = init_session(self.model_path)

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {"model_path": self.model_path}

    def __setstate__(self, values):
        self.model_path = values["model_path"]
        self.sess = init_session(self.model_path)
