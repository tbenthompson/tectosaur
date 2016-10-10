from tectosaur.table_lookup import *

# Most of the tests for the table lookup are indirect through the integral op tests,
# these are just for specific sub-functions

def test_adjacent_theta():
    rho = 0.2
    for i in range(10):
        theta = np.random.rand(1)[0] * 2 * np.pi
        out = get_adjacent_theta(
            np.array([[0,0,0],[1,0,0],[0.5,rho,0.0]]),
            np.array([[0,0,0],[1,0,0],[0.5,rho*np.cos(theta),rho*np.sin(theta)]]),
        )
        np.testing.assert_almost_equal(out, theta)
