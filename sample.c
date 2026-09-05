int func(int xyz)
{
    xyz=xyz+1;
    return xyz;
}

int main()
{
    int a, b=5;
    for(int i=0;i<=10;i++)
    {
        b=b+func(i);
    }
    return 0;
}