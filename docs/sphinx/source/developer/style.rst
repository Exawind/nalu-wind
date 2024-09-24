Nalu Style Guide
================

1. No tabs. Remove them from your editor. Better yet, use eclipse and follow the xml style. Use the format `here <https://github.com/exawind/nalu-wind/blob/master/SQA/naluEclipseFormat.xml>`__.

2. Use underscores for private data, e.g., ``const double thePrivateData_``.

3. Use camel case for data members and classes unless it is silly (you get the idea).

4. Camel case on Class names always; non camel case for methods, e.g.,

::

   const double Realm::get_me() {
     return hereIAm_; // hmmm... silly? your call
   }

5. Use ``const`` when possible, however, do not try to be a member of the 'const' police force. 

6. We need logic to launch some special physics. Try to avoid run time logic by designing with polymorphic/templates.

7. When possible, add classes that manage loading, field registration, setup and execute, e.g., SolutionNormPostProcessing, etc.


