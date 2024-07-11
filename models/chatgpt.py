import time
import openai

class ChatGPT():
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, message, temperature=0.7, max_tokens=4096, endpoint='gpt-3.5-turbo-16k'):

        messages = [ {"role": "system", "content": "You are an intelligent and helpful assistant. You follow all the instructions in the user prompt, and provide all the questions and answers they ask for."} ]

        # message = input("User : ")
        # if message:
        messages.append(
            {"role": "user", "content": message},
        )

        while True:
            try:
                # Place the call you're trying to make here
                chat = self.client.chat.completions.create(model=endpoint,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens)
                break  # Break the loop if the request was successful
            except openai.APITimeoutError:
                print('Request timed out. Retrying...')
                time.sleep(3)
                continue
            except openai.APIConnectionError:
                print('API connection error. Retrying...')
                time.sleep(3)
                continue
            except openai.InternalServerError:
                print(f'Internal server error. Retrying...')
                time.sleep(3)
                continue
            except openai.APIError as e:
                print('Their error. Retrying...')
                time.sleep(3)
                continue
            # except openai.ServiceUnavailableError:
            #     print(f'Service unavailable. Retrying...')
            #     time.sleep(3)
                # continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise  # This will re-raise the caught exception, effectively stopping your program

        reply = chat.choices[0].message.content

        return reply
        # print(f"ChatGPT: {reply}")
        # messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    # chatgpt_generate('If I have three lemons and I get an extra two, how many did I have before?')
    chatgpt_generate('''  Here is a schema for annotating retinal OCT images.

        Attributes are normal or absent (N) by default.
        - Drusen (Y/N, size: small/medium/large, confluent or not)
        - Pigment epithelial detachment (PED) (Y/N, type: serous/drusenoid, size)
        - Shallow irregular RPE elevation (SIRE) (Y/N, size of RPE covered, double-layer sign or not)
        - RPE state (normal/degeneration/atrophy, outer retinal layer atrophy or not)
        - RPE disruption/elevation (Y/N)
        - Subretinal fluid (Y/N, volume: small/moderate/large)
        - Intraretinal fluid (Y/N, volume: small/moderate/large)
        - Intraretinal hyperreflective foci (Y/N, number)
        - Signal hypertransmission to the choroid (Y/N, severity: mild/moderate/significant, optional types: cRORA/iRORA)
        - Subretinal hyperreflective material (SHRM) (Y/N)
        - Pathology unrelated to AMD (Y/N)
        - Image quality (good unless specified, specify if prevents detailed analysis)
        - Overall impression of AMD stage determined from the other attributes (early, intermediate, wet, late, previously treated wet AMD, normal ageing)

    Here are some annotations of images generated with reference to the schema:

    179	This OCT image shows shallow irregular RPE elevation from the nasal macula to the temporal macula. There is evidence of a double layer-sign visible in the centre of the image. There is no intraretinal or subretinal fluid present and no intraretinal hyperreflective foci. There is some generalised increase in signal hypertransmission to the choroid suggesting RPE degeneration. This image is consistent with intermediate AMD.
    180	This OCT image shows a moderate volume of intraretinal fluid with one large cyst in the centre of the image and smaller cysts surrounding it. There is some evidence of hyperreflectivity at the level of the photoreceptors beneath the intraretinal fluid. There is no subretinal fluid and there are no drusen. There is no significant increase in signal transmission to the choroid. This image may be consistent with wet AMD.
    181	This OCT image shows a large area of shallow irregular RPE elevation with a double layer sign visible that extends across the majority of the image. There is no intraretinal or subretinal fluid and no hyperreflective foci. There is some evidence of increased signal hypertransmission to the choroid in the right of the image. This image is consistent with intermediate AMD.
    182	This OCT image shows a moderate amount of subretinal hyperreflective material, mainly in the centre of the image, suggesting scarring. There is extensive signal hypertransmission to the choroid indicating widespread RPE atrophy. There is no intraretinal or subretinal fluid and no intraretinal hyperreflective foci. This image is consistent with previously treated wet AMD.
    183	This OCT image shows extensive subretinal hyperreflective material across the extent of the image likely to be consistent with scarring. There is a small hyporeflective area beneath the scarring in the left of the image which may be artefact or may be a small volume of subretinal fluid. There is no intraretinal fluid. There is moderately increased signal hypertransmission to the choroid across the image. This image is consistent with previously treated wet AMD.
    184	This OCT image is poor quality which affects interpretation but appears to show a number of medium sized drusen across the image with extensive RPE and outerretinal layer atrophy between the drusen. There is no obvious intraretinal or subretinal fluid. This image may be consistent with late stage dry AMD.
    185	This OCT image shows evidence of small or medium drusen in the centre and right of the image. There is no intraretinal or subretinal fluid and no intraretinal hyperreflective foci. There is no increased signal transmission to the choroid. This image may be consistent with early or intermediate AMD.
    186	This OCT image shows an area of shallow irregular RPE elevation from the left to the centre of the image with evidence of a double-layer sign beneath. There is no evidence of intraretinal or subretinal fluid and no intraretinal hyperreflective foci. There is mildly increased signal hypertransmission to the choroid mainly in the centre of the image. This image may be consistent with intermediate AMD.
    187	This OCT image shows two large central drusenoid PEDs (pigment epithelial detachments) with medium sized drusen in the left and righ tof the image. There is no intraretinal or subretinal fluid, no intraretinal hyperreflective foci and minimal signal hypertransmission to the choroid. This image is consistent with intermediate AMD.
    188	This OCT image shows a large area of irregular RPE elevation extending across the majority of the image. In the right of the image there is a small volume of subretinal fluid. There is no intraretinal fluid. There are no intraretinal hyperreflective foci and no significant signal hypertransmission to the choroid. This image is consistent with wet AMD.
    189	This OCT image shows three medium or large drusen in the centre of the image. The druse on the right side has a hyporeflective core and there is evidence of a small volume of intraretinal fluid above it. There is no subretinal fluid, no intraretinal hyperreflective foci and minimal signal hypertransmission to the choroid. This image may be consistent with wet AMD.
    190	This OCT image appears to show a thickened retina in the centre of the image but no specific evidence of AMD. 
    191	This OCT image shows small or medium drusen across the image. There is no intraretinal or subretinal fluid. There may be one small intraretinal hyperreflective focus above a druse in the right of the image. There is no signal hypertransmission to the choroid. This image is consistent with early or intermediate AMD.
    192	This OCT image shows an area of shallow irregular RPE elevation (SIRE) in the left of the image extending to the centre. There is a small volume of intraretinal fluid visible above this and a double-layer sign visible beneath. There are small or medium sized drusen to the right of the SIRE. There is no intraretinal fluid and no intraretinal hyperreflective foci. There is no increased signal hypertransmission to the choroid. This OCT image is consistent with wet AMD.

    Below is the sample table documenting that data:

    Image  | Drusen | Drusen Size | Drusen Confluence | PED | PED Type | PED Size | SIRE | SIRE Size | Double-Layer Sign | RPE State | RPE Disruption/Elevation | Subretinal Fluid | Subretinal Fluid Volume | Intraretinal Fluid | Intraretinal Fluid Volume | Intraretinal Hyperreflective Foci | Signal Hypertransmission | Signal Hypertransmission Severity | SHRM | Pathology Unrelated to AMD | Image Quality | Overall AMD Stage
    ------ | ------ | ---------- | ----------------- | --- | -------- | -------- | ---- | --------- | ----------------- | --------- | ---------------------- | ---------------- | --------------------- | ----------------- | ---------------------- | --------------------------- | --------------------- | ----------------------------- | ---- | ----------------------- | ------------ | ----------------
    179    | N      | N          | N                 | N   | N        | N        | Y    | Large     | Y (Central)        | N         | Y (Mild)               | N                | N                     | N                 | N                       | N                           | Y (Generalized Increase) | Significant                | N    | N                       | Good         | Intermediate
    180    | N      | N          | N                 | N   | N        | N        | N    | N         | N                  | N         | N                       | N                | N                     | Y (Moderate Volume) | Large                  | Y (Photoreceptor Level)     | N                    | N                           | N    | N                       | Good         | Wet AMD
    181    | N      | N          | N                 | N   | N        | N        | Y    | Large     | Y (Extends)         | N         | N                       | N                | N                     | N                 | N                       | N                           | Y (Some Evidence)    | N                           | N    | N                       | Good         | Intermediate
    182    | N      | N          | N                 | N   | N        | N        | N    | N         | N                  | N         | Y (Moderate)            | N                | N                     | N                 | N                       | N                           | Y (Extensive)        | Significant                | Y    | N                       | Good         | Previously Treated Wet AMD
    183    | N      | N          | N                 | N   | N        | N        | N    | N         | N                  | N         | N                       | Y (Small Volume) | N                     | N                 | N                       | N                           | N                    | Moderate                    | Y    | N                       | Good         | Previously Treated Wet AMD
    184    | Y      | Medium     | N                 | N   | N        | N        | N    | N         | N                  | Y (Outer Retinal Layer Atrophy) | N             | N                | N                     | N                 | N                       | N                           | N                    | N                           | N    | N                       | Poor (Quality affects interpretation) | Late Stage Dry AMD
    185    | Y      | Small/Medium | N              | N   | N        | N        | N    | N         | N                  | N         | N                       | N                | N                     | N                 | N                       | N                           | N                    | N                           | N    | N                       | Good         | Early/Intermediate AMD
    186    | N      | N          | N                 | N   | N        | N        | Y    | Mild      | Y (Extending)       | N         | N                       | N                | N                     | N                 | N                       | N                           | Y (Mild)             | Mild                        | N    | N                       | Good         | Intermediate
    187    | N      | N          | N                 | Y   | Drusenoid| Large    | N    | N         | N                  | N         | N                       | N                | N                     | N                 | N                       | N                           | Y (Minimal)          | Minimal                     | N    | N                       | Good         | Intermediate
    188    | N      | N          | N                 | N   | N        | N        | Y    | Large     | Y (Small Volume)    | N         | N                       | Y (Small Volume) | Large                | N                 | N                       | N                           | N                    | N                           | N    | N                       | Good         | Wet AMD
    189    | Y      | Medium/Large | N             | N   | N        | N        | N    | N         | N                  | N         | N                       | N                | N                     | Y                 | Small                    | N                           | N                    | N                           | N    | N                       | Good         | Wet AMD
    190    | N      | N          | N                 | N   | N        | N        | N    | N         | N                  | N         | N                       | N                | N                     | N                 | N                       | N                           | N                    | N                           | N    | N                       | Poor (No specific evidence of AMD) | N/A
    191    | Y      | Small/Medium | N              | N   | N        | N        | N    | N         | N                  | N         | N                       | N                | N                     | N                 | N                       | Y (Small)                    | N                    | N                           | N    | N                       | Good         | Early/Intermediate AMD
    192    | Y      | Small/Medium | N              | Y   | Drusenoid| Small    | Y    | Mild      | Y (Central)         | N         | N                       | N                | N                     | Y                 | Small                    | N                           | N                    | N                           | N    | N                       | Good         | Wet AMD


    Here are some new image annotations that were guided by the schema:

    1	This is an OCT image which shows a large drusenoid PED (pigment epithelial detachment) with a hyporeflective core and a moderate amount of subretinal hyperreflective material (SHRM). There is moderate hypertransmission beneath the SHRM. There is a small amount of subretinal fluid between the PED and the SHRM. There are also a small number of intraretinal hyperreflective foci.
    2	This OCT image shows a double layer sign with some moderately reflective material between Bruch's membrane and the RPE. There is also a small amount of subretinal hyperreflective material above the area with the double layer sign. In the rest of the image there is some mild RPE degeneration with slightly increased transmission as a result but no obvious drusen or fluid.
    3	This OCT image shows several medium and large drusen. The majority of which have hyporeflective cores but one has a hyperreflective core. There are several hyporeflective cystic shaped areas within the retina suggestive of intraretinal fluid at the fovea. There is no obvious choroidal hypertransmission and there are no intraretinal hyperreflective foci. There is no significant subretinal fluid.
    4	This OCT image shows a healthy retina with no obvious signs of AMD.
    5	This is an OCT image which shows a moderate amount of subretinal hyperreflective material (SHRM) with some choroidal hypertransmission of the signal beneath it, suggestive of outer retina and RPE atrophy. There are no obvious drusen and there is no intraretinal or subretinal fluid.
    6	This OCT image shows a moderate-sized area of fibrovascular pigment epithelial detachment (PED) with a moderate amount of subretinal hyperreflective material between the PED and the retina. There are areas of intermittent signal hypertransmission into the choroid in the area of the PED suggesting overlying outer retina and RPE atrophy. There is also one area of cystic-shaped hyporeflectivity within the retina indicating a small amount of intraretinal fluid.
    7	This OCT image shows an area at the fovea with loss of the ellipsoid zone in the outer retina, degenerative change to the RPE and some signal hypertransmission to the choroid. This is suggestive of iRORA. There is no intraretinal fluid or subretinal fluid. There are no significant intraretinal hyperreflective foci.
    8	This OCT image shows an moderately-sized area of subretinal fluid overlying an area containing medium-sized drusen with a small amount of subretinal hyperreflective material (SHRM). There is also a small amount of signal hypertransmission to the choroid in this area too. There is no intraretinal fluid. 
    9	In this OCT image there are no drusen and no areas of subretinal fluid or intraretinal fluid. There is an area in the central part of the image where ellipsoid zone is less clear but there is no signal hypertransmission beneath this and no other signs of AMD.
    10	This OCT image shows a moderate-sized area of intra-retinal fluid in the inner retina. There is no subretinal fluid and there are no drusen and no signal transmission defects.

    I want you to supply new rows based on the provided annotations. Rewrite the column headings but please only write the new rows.''')