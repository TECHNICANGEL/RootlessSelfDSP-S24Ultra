//
// Created by tim on 01.07.22.
//

#include "JArrayList.h"

#define TAG "JArrayList_JNI"
#include <Log.h>

JArrayList::JArrayList(JNIEnv* env) : IJavaObject(env)
{
    jclass localArrayClass = _env->FindClass("java/util/ArrayList");
    if (localArrayClass == nullptr)
    {
        LOGE("JArrayList::ctor: java/util/ArrayList class not found");
        return;
    }

    jmethodID methodInit = _env->GetMethodID(localArrayClass, "<init>", "()V");
    if (methodInit == nullptr)
    {
        LOGE("JArrayList::ctor: java/util/ArrayList<init>()V method not found");
        // MEMORY LEAK FIX: Delete LocalRef on early return
        _env->DeleteLocalRef(localArrayClass);
        return;
    }

    innerArrayList = _env->NewObject(localArrayClass, methodInit);
    if (innerArrayList == nullptr)
    {
        LOGE("JArrayList::ctor: Failed to allocate ArrayList object");
        _env->DeleteLocalRef(localArrayClass);
        return;
    }

    methodAdd = _env->GetMethodID(localArrayClass, "add", "(Ljava/lang/Object;)Z");
    if (methodAdd == nullptr)
    {
        LOGE("JArrayList::ctor: java/util/ArrayList.add(Ljava/lang/Object;)Z method not found");
        _env->DeleteLocalRef(localArrayClass);
        return;
    }

    // MEMORY LEAK FIX: Delete LocalRef after extracting method IDs
    _env->DeleteLocalRef(localArrayClass);

    _isValid = true;
}

bool JArrayList::isValid() const {
    return _isValid;
}

bool JArrayList::add(jobject object) {
    return _env->CallBooleanMethod(innerArrayList, methodAdd, object);
}

jobject JArrayList::getJavaReference() {
    return innerArrayList;
}
