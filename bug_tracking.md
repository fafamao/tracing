1. Unmatching parameter cause kernel to exit early without any indication
   For example, when objects are created, parameters are passed as double but the class accepts float. It makes the construction fail and kernel scene generation fails.
