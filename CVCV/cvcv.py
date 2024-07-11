import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def traiter_fichiers(descriptions_de_poste, cvs):
    textes_descriptions = []
    textes_cv = []

    for description in descriptions_de_poste:
        with pdfplumber.open(description) as pdf:
            texte_description = " ".join([page.extract_text() for page in pdf.pages])
            textes_descriptions.append(texte_description)

    for cv in cvs:
        with pdfplumber.open(cv) as pdf:
            texte_cv = " ".join([page.extract_text() for page in pdf.pages])
            textes_cv.append(texte_cv)

    correspondances = []

    for texte_description in textes_descriptions:
        for texte_cv in textes_cv:
            contenu = [texte_description, texte_cv]
            vecteur_compte = CountVectorizer()
            matrice = vecteur_compte.fit_transform(contenu)
            matrice_similarite = cosine_similarity(matrice)
            correspondance = matrice_similarite[0][1] * 100
            correspondances.append((texte_description, texte_cv, correspondance))

    return correspondances

def about_us():
    st.title("ABOUT US")
    st.subheader("WELCOME TO ABOUT US PAGE")
    st.write("Meet the creators of this application.")
    creators_info = {
        "Mohamed Elhaimer": "Co-fondateur et développeur",
    }

    for creator, description in creators_info.items():
        st.write(f"**{creator}** : {description}")

def contact_us():
    st.title("CONTACT US")
    st.subheader("Use the form below to contact us")
    name = st.text_input("Your name :")
    email = st.text_input("Your email :")
    message = st.text_area("Your message :")

    if st.button("submit"):
        st.success("Form submitted successfully! We will respond to you soon")

def main():
    nav_choice = st.sidebar.radio("Navigation", ["HOME", "Upload CV", "ABOUT US", "CONTACT US"])

    if nav_choice == "HOME":
        st.title("CVCV")
        st.write("Bienvenue sur l'application CVCV! Cet outil est conçu pour analyser les CV et les descriptions de poste "
                 "pour déterminer le pourcentage de correspondance entre un candidat et un emploi.")
        st.write("L'objectif principal est de vérifier si un candidat est qualifié pour un poste en fonction de ses "
                 "études, de son expérience et d'autres informations capturées dans son CV. Il utilise des techniques "
                 "de traitement du langage naturel (NLP) pour la correspondance de motifs.")

        col1, col2 = st.columns(2)
        with col2:
            st.image("C:/Users/HP/Desktop/CV.CV/CVCV/images/home.png", width=400, output_format="PNG")

    elif nav_choice == "Upload CV":
        st.subheader("CVCV ANALYSER")
        uploadedJD = st.file_uploader("Upload job Description (PDF)", accept_multiple_files=True, type="pdf")
        uploadedResume = st.file_uploader("Upload CV (PDF)", accept_multiple_files=True, type="pdf")

        click = st.button("Process")

        if click and uploadedJD and uploadedResume:
            correspondances = traiter_fichiers(uploadedJD, uploadedResume)

            for i, (texte_description, texte_cv, correspondance) in enumerate(correspondances):
                correspondance = round(correspondance, 2)
                st.write(f"Correspondance {i+1} : {correspondance}%")

                if correspondance >= 70:
                    st.success("Félicitations! Le candidat est un bon match pour le poste.")
                else:
                    st.warning("Le candidat peut ne pas être un match idéal pour le poste. Envisagez de revoir les critères.")

                st.subheader("Texte Extrait:")
                st.write("Description du Poste:")
                st.text(texte_description)
                st.write("CV:")
                st.text(texte_cv)

    elif nav_choice == "ABOUT US":
        about_us()

    elif nav_choice == "CONTACT US":
        contact_us()

if __name__ == "__main__":
    main()
