"""Training side of lrl (LRL_PLAN.md section 6): stage simulator and training scenes.

Nothing here is imported by the harness at run time. Simulator truth (joint values of scene parts, contacts) is
read here for labels, rewards and scoring only; a skill's inputs stay those of atrace.sim_iface.SimRobot.
"""
