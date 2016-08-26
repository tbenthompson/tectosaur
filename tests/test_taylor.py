def test_taylor():
    import cppimport
    test_taylor = cppimport.imp("tectosaur.test_taylor").test_taylor
    test_taylor.run_tests([])
