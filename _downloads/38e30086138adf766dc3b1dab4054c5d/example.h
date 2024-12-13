/** @file example.h
 *  @brief Brief description of a documented file.
 *
 *  Longer description of a documented file.
 */

/** Here is a brief description of the example class.
 *
 * This is a more in-depth description of the class.
 * This class is meant as an example.
 * It is not useful by itself, rather its usefulness is only a
 * function of how much it helps the reader. It is in a sense
 * defined by the person who reads it and otherwise does
 * not exist in any real form.
 *
 * @note This is a note.
 *
 */

#ifndef EXAMPLECLASS_H
#define EXAMPLECLASS_H

class ExampleClass
{

public:
  /// Create an ExampleClass.
  ExampleClass();

  /** Create an ExampleClass with lot's of intial values.
   *
   * @param a This is a description of parameter a.
   * @param b This is a description of parameter b.
   *
   * The distance between \f$(x_1,y_1)\f$ and \f$(x_2,y_2)\f$ is
   * \f$\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}\f$.
   */
  ExampleClass(int a, float b);

  /** ExampleClass destructor description.
   */
  ~ExampleClass();

  /// This method does something.
  void DoSomething();

  /**
   * This is a method that does so
   * much that I must write an epic
   * novel just to describe how much
   * it truly does.
   */
  void DoNothing();

  /** Brief description of a useful method.
   * @param level An integer setting how useful to be.
   * @return Description of the output.
   *
   * This method does unbelievably useful things.
   * And returns exceptionally useful results.
   * Use it everyday with good health.
   * \f[
   *    |I_2|=\left| \int_{0}^T \psi(t)
   *        \left\{
   *           u(a,t)-
   *           \int_{\gamma(t)}^a
   *           \frac{d\theta}{k(\theta,t)}
   *           \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
   *        \right\} dt
   *     \right|
   * \f]
   */
  void* VeryUsefulMethod(bool level);

  /** Brief description of a useful method.
   * @param level An integer setting how useful to be.
   * @return Description of the output.
   *
   * - Item 1
   *
   *   More text for this item.
   *
   * - Item 2
   *   + nested list item.
   *   + another nested item.
   * - Item 3
   *
   * # Markdown Example
   * [Here is a link.](http://www.google.com/)
   */
  void* AnotherMethod(bool level);

protected:
  /** The protected methods can be documented and extracted too.
   *
   */
  void SomeProtectedMethod();

private:
  const char* fQuestion; ///< The question
  int fAnswer;           ///< The answer

}; // End of class ExampleClass

#endif // EXAMPLE_H
