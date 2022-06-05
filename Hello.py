import streamlit as st

# ADDING A TITLE AND FAVICON
st.set_page_config(page_title="Grant Calculator", page_icon="🦖")


def intro():
    st.write("# Welcome to Grant Calculator! 👋")

    st.markdown(
        """
        Grant Calculator estimates potential grant amounts. Use it to see how individual
        parameters affect estimates!

        **👈 Select how many baseline will be replaced on the left** to go to the applicable
        calculator!

        ### Want to learn more?

        - Check out [Carl Moyer Memorial Air Quality Standards Attainment Program (Carl 
          Moyer Program)](https://ww2.arb.ca.gov/our-work/programs/carl-moyer-memorial-air-quality-standards-attainment-program)
        - Jump into the [Carl Moyer Program requirements](https://ww2.arb.ca.gov/guidelines-carl-moyer)
    """
    )

    st.markdown(
        """
        ### Dislcaimer

        While we have made every attempt to ensure that the information contained in this 
        site has been obtained from reliable sources, we are not responsible for any errors
        or omissions, or for the results obtained from the use of this information. All 
        information in this site is provided "as is", with no guarantee of completeness,
        accuracy, timeliness or of the results obtained from the use of this information,
        and without warranty of any kind, express or implied, including, but not limited
        to warranties of performance, merchantability and fitness for a particular purpose.

        Certain links in this site connect to other websites maintained by third parties
        over whom we have no control. We make no representations as to the accuracy or
        any other aspect of information contained in other websites."""
    )


def main():
    intro()


if __name__ == "__main__":
    main()
