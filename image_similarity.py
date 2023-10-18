import cv2


def image_similarity(img1, img2):
    # nfeatures=2000
    # nfeatures=20
    orb = cv2.ORB_create()

    # Resize
    # sizeDefault = (1920, 1080)
    sizeDefault = (1200, 600)
    img1Res = cv2.resize(img1, sizeDefault)
    img2Res = cv2.resize(img2, sizeDefault)

    # Esse é o ponto onde serão dectados pontos chaves e descritores das imagens
    kp_a, desc_a = orb.detectAndCompute(img1Res, None)
    kp_b, desc_b = orb.detectAndCompute(img2Res, None)

    # Definir o objeto combinador de força bruta.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Nessa etapa será feito o match das imagens, ou seja, a combinação das imagens
    matches = bf.match(desc_a, desc_b)

    # Aqui será feito a procura por regiões similares das imagens
    #   com a distância menor que 50.
    # É possível ajustar a distância entre 0 a 100,
    #   sendo 0 as imagens totalmente diferente e 100 as imagens iguais
    # Nesse caso é escolhido a distância de 50.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)
